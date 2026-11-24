// TigerStyle Inference Engine: single-arena runtime VRAM, 48-visit Blackwell forward pass,
// CUDA Graphs (B=1 decode, speculative B=2..8, prefill B=8), aux_stream DSpark overlap,
// zero-allocation streaming token generation, and long-context benchmark.

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const tiger = @import("tiger.zig");
const vram_arena = @import("vram_arena.zig");
const weights_mod = @import("weights.zig");

pub const TokenCallbackFn = *const fn (context: *anyopaque, token_id: i32) bool;

pub const SampleParams = struct {
    temperature: f32 = 0.0,
    top_p: f32 = 0.95,
    top_k: i32 = 0,
    seed: u64 = 0,
};

pub const GenerateStats = struct {
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    prefill_ms: f64 = 0.0,
    decode_ms: f64 = 0.0,
    spec_steps: u32 = 0,
    lookup_steps: u32 = 0,
    total_drafts: u32 = 0,
    total_accepted: u32 = 0,
    last_token_id: i32 = -1,
};

/// Zero-malloc O(1) n-gram prompt-lookup drafter (inspired by Atomic Engine's hybrid lookup + DSpark fallback).
/// Keeps a recent 8,192-token ring and a 16,384-slot two-entry hash index (~320 KiB host RAM, 0 VRAM).
/// On a >=16-token exact suffix match, proposes the next 7 tokens directly to the B=8 verify graph,
/// bypassing the DSpark neural pass completely (~5 ns CPU lookup vs ~400 us GPU draft).
const PromptLookupIndex = struct {
    const hist_cap: usize = 8_192;
    const hist_mask: usize = hist_cap - 1;
    const table_cap: usize = 16_384;
    const table_mask: usize = table_cap - 1;
    const min_match_len: u32 = 16;

    hist: [hist_cap]i32 = @splat(0),
    count: u32 = 0,
    gen: u16 = 0,
    table_hash: [table_cap]u64 = @splat(0),
    table_pos0: [table_cap]u32 = @splat(0),
    table_pos1: [table_cap]u32 = @splat(0),
    table_gen: [table_cap]u16 = @splat(0),

    fn reset(self: *PromptLookupIndex) void {
        self.count = 0;
        self.gen +%= 1;
        if (self.gen == 0) {
            @memset(&self.table_gen, 0);
            self.gen = 1;
        }
    }

    inline fn hash4(a: i32, b: i32, c: i32, d: i32) u64 {
        var h: u64 = @as(u64, @bitCast(@as(i64, a))) *% 0x9E3779B97F4A7C15;
        h = (h ^ @as(u64, @bitCast(@as(i64, b)))) *% 0xBF58476D1CE4E5B9;
        h = (h ^ @as(u64, @bitCast(@as(i64, c)))) *% 0x94D049BB133111EB;
        h = (h ^ @as(u64, @bitCast(@as(i64, d)))) *% 0x9E3779B97F4A7C15;
        return h ^ (h >> 31);
    }

    inline fn at(self: *const PromptLookupIndex, abs_idx: u32) i32 {
        return self.hist[abs_idx & hist_mask];
    }

    fn push(self: *PromptLookupIndex, tok: i32) void {
        const idx = self.count;
        self.hist[idx & hist_mask] = tok;
        self.count = idx + 1;
        if (self.count >= 4) {
            const h = hash4(self.at(self.count - 4), self.at(self.count - 3), self.at(self.count - 2), self.at(self.count - 1));
            const slot: usize = @intCast(h & table_mask);
            if (self.table_gen[slot] == self.gen and self.table_hash[slot] == h) {
                self.table_pos1[slot] = self.table_pos0[slot];
                self.table_pos0[slot] = self.count;
            } else {
                self.table_hash[slot] = h;
                self.table_pos0[slot] = self.count;
                self.table_pos1[slot] = 0;
                self.table_gen[slot] = self.gen;
            }
        }
    }

    fn match(self: *const PromptLookupIndex, out_drafts: *[constants.mtp_draft_count_max]i32) bool {
        const n = self.count;
        if (n < min_match_len + constants.mtp_draft_count_max + 1) return false;
        const h = hash4(self.at(n - 4), self.at(n - 3), self.at(n - 2), self.at(n - 1));
        const slot: usize = @intCast(h & table_mask);
        if (self.table_gen[slot] != self.gen or self.table_hash[slot] != h) return false;

        const cand: u32 = if (self.table_pos0[slot] == n) self.table_pos1[slot] else self.table_pos0[slot];
        if (cand < min_match_len) return false;
        if (cand + constants.mtp_draft_count_max >= n) return false;
        const win_lo: u32 = if (n > hist_cap) (n - @as(u32, @intCast(hist_cap))) else 0;
        if (cand - min_match_len < win_lo) return false;

        var k: u32 = 1;
        while (k <= min_match_len) : (k += 1) {
            if (self.at(cand - k) != self.at(n - k)) return false;
        }
        var j: u32 = 0;
        while (j < constants.mtp_draft_count_max) : (j += 1) {
            out_drafts[j] = self.at(cand + j);
        }
        return true;
    }
};

const prefix_checkpoint_count: usize = 4;
const side_shield_max_tokens: u32 = 4096;
const prefill_batch_max: u32 = 2048;
const prefill_local_attn_sub: u32 = 256;
const prefill_sub_chunks_max: u32 = prefill_batch_max / constants.mtp_block_size_max; // 256

inline fn offset_f16_ptr(base: *anyopaque, elem_offset: usize) *anyopaque {
    const bytes: [*]u8 = @ptrCast(base);
    return @ptrCast(bytes + elem_offset * 2);
}

const LocalStateCheckpoint = struct {
    pos: u32 = 0,
    valid: bool = false,
    data: []u8 = &.{},
};

const PrefixCache = struct {
    enabled: bool = false,
    local_state_bytes: usize = 0,
    session_tokens: []i32 = &.{},
    session_len: u32 = 0,
    session_use_mtp: bool = false,
    gpu_live_at_pos: i32 = -1,
    checkpoints: [prefix_checkpoint_count]LocalStateCheckpoint = @splat(.{}),
    side_shield_buf: []u8 = &.{},
};

pub const Engine = struct {
    weights: ?*const weights_mod.ModelWeights = null,
    context_tokens_max: u32 = 0,
    use_graphs: bool = true,
    kv_format: constants.KVFormat = .fp8,
    share_loop_kv: bool = false,
    current_position: i32 = 0,
    schedule: [constants.visit_count]constants.VisitInfo = undefined,

    arena: vram_arena.VRAMArena = .{},
    stream: ?*anyopaque = null,
    aux_stream: ?*anyopaque = null,
    fork_event: ?*anyopaque = null,
    join_event: ?*anyopaque = null,
    done_event: ?*anyopaque = null,

    // Cached CUDA Graphs
    b1_graph_exec: ?*anyopaque = null,
    spec_graph_exec: [constants.mtp_draft_count_max + 1]?*anyopaque = @splat(null),
    draft_graph_exec: ?*anyopaque = null,
    verify_b8_graph_exec: ?*anyopaque = null,
    adaptive_graph: ?*anyopaque = null,
    adaptive_graph_threshold: f32 = -1.0,
    spec_threshold: f32 = 0.80,
    use_prompt_lookup: bool = true,
    prefill_b8_graph_exec: [2]?*anyopaque = .{ null, null },

    // Pinned host readback/upload buffers (zero-malloc decode loop)
    h_accepted: ?[*]i32 = null,
    h_num_accepted: ?*i32 = null,
    h_chosen_width: ?*i32 = null,
    h_lookup_drafts: ?[*]i32 = null,

    // Multi-checkpoint host-RAM prefix KV cache (zero extra VRAM!)
    prefix_cache: PrefixCache = .{},
    prompt_lookup: PromptLookupIndex = .{},

    // Per-visit KV caches and sconv rings
    kv_cap: [constants.visit_count]i32 = @splat(0),
    k_cache: [constants.visit_count]?*anyopaque = @splat(null),
    v_cache: [constants.visit_count]?*anyopaque = @splat(null),
    k_scale: [constants.visit_count]?*anyopaque = @splat(null),
    v_scale: [constants.visit_count]?*anyopaque = @splat(null),
    k_ring: [constants.visit_count][*]f32 = undefined,
    v_ring: [constants.visit_count][*]f32 = undefined,
    attn_ring: [constants.visit_count][*]f32 = undefined,
    mlp_ring: [constants.visit_count][*]f32 = undefined,

    // DSpark KV caches and f_ring
    ds_kv_cap: [constants.mtp_layer_count]i32 = .{ 0, 0, 0 },
    ds_k_cache: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_v_cache: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_k_scale: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_v_scale: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_f_ring: ?[*]f32 = null,

    // Activation scratch arena (sized for B = mtp_block_size_max = 8)
    split_buf: [*]f32 = undefined,
    d_pos: *i32 = undefined,
    d_tokens: [*]i32 = undefined,
    d_next_tokens: [*]i32 = undefined,
    d_anchor: *i32 = undefined,
    d_drafts: [*]i32 = undefined,
    d_draft_conf: [*]f32 = undefined,
    d_chosen_width: *i32 = undefined,
    d_accepted: [*]i32 = undefined,
    d_num_acc: *i32 = undefined,

    hidden: [*]f32 = undefined,
    prelude: [*]f32 = undefined,
    partial: [*]f32 = undefined,
    blocks: [*]f32 = undefined,
    saved: [*]f32 = undefined,
    prior_last: [*]f32 = undefined,
    attn_in: [3][*]f32 = undefined,
    attn_in_h: [3]*anyopaque = undefined,

    qrgs: [*]f32 = undefined,
    k_proj: [*]f32 = undefined,
    v_proj: [*]f32 = undefined,
    attn_gated: [*]f32 = undefined,
    attn_gated_h: *anyopaque = undefined,
    o_proj: [*]f32 = undefined,
    residual: [*]f32 = undefined,
    mlp_in: [*]f32 = undefined,
    mlp_normed: [*]f32 = undefined,
    mlp_normed_h: *anyopaque = undefined,
    ctrl_msg: [*]f32 = undefined,
    ctrl_cat: [*]f32 = undefined,
    ctrl_d: [*]f32 = undefined,
    router_logits: [*]f32 = undefined,
    route_ids: [*]i32 = undefined,
    route_w: [*]f32 = undefined,
    exp_groups: *anyopaque = undefined,
    moe_act_h: *anyopaque = undefined,
    moe_down: [*]f32 = undefined,
    moe_slot_out: [*]f32 = undefined,
    taps: [*]f32 = undefined,
    ds_mix: [*]f32 = undefined,
    ds_kv: [*]f32 = undefined,
    ds_ws: [*]f32 = undefined,
    lm_blk_val: [*]f32 = undefined,
    lm_blk_idx: [*]i32 = undefined,
    lm_logits: [*]f32 = undefined,
    d_draft_probs: [*]f32 = undefined,
    d_draft_q_sel: [*]f32 = undefined,
    d_draft_is_delta: *i32 = undefined,
    d_sample_cfg: *cuda.CSampleConfig = undefined,
    sample_params: SampleParams = .{},

    // Fast batched prefill scratch (B <= 256, ~119 MiB total inside same VRAMArena)
    p_tokens: [*]i32 = undefined,
    p_pos_vec: [*]i32 = undefined,
    p_hidden: [*]f32 = undefined,
    p_prelude: [*]f32 = undefined,
    p_partial: [*]f32 = undefined,
    p_blocks: [*]f32 = undefined,
    p_saved: [*]f32 = undefined,
    p_prior_last: [*]f32 = undefined,
    p_attn_in: [3][*]f32 = undefined,
    p_attn_in_h: [3]*anyopaque = undefined,
    p_qrgs: [*]f32 = undefined,
    p_k_proj: [*]f32 = undefined,
    p_v_proj: [*]f32 = undefined,
    p_attn_gated_h: *anyopaque = undefined,
    p_o_proj: [*]f32 = undefined,
    p_residual: [*]f32 = undefined,
    p_mlp_in: [*]f32 = undefined,
    p_mlp_normed: [*]f32 = undefined,
    p_mlp_normed_h: *anyopaque = undefined,
    p_router_logits: [*]f32 = undefined,
    p_route_ids: [*]i32 = undefined,
    p_route_w: [*]f32 = undefined,
    p_exp_offsets: [*]i32 = undefined,
    p_sorted_tok: [*]i32 = undefined,
    p_sorted_slot: [*]i32 = undefined,
    p_sorted_w: [*]f32 = undefined,
    p_moe_act_h: *anyopaque = undefined,
    p_moe_slot_out: [*]f32 = undefined,
    p_taps: [*]f32 = undefined,
    p_w_f16: *anyopaque = undefined,
    p_ctrl_msg: [*]f32 = undefined,
    p_ctrl_cat_h: *anyopaque = undefined,
    p_ctrl_d: [*]f32 = undefined,
    p_ctrl_d_h: *anyopaque = undefined,
    p_ctrl_f: [*]f32 = undefined,
    p_ctrl_g: [*]f32 = undefined,

    pub fn init(
        self: *Engine,
        weights: *const weights_mod.ModelWeights,
        context_tokens_max: u32,
        use_graphs: bool,
        kv_format: constants.KVFormat,
        share_loop_kv_req: bool,
    ) void {
        tiger.assert(context_tokens_max >= 64);
        tiger.assert(self.arena.base_ptr == null);

        self.weights = weights;
        self.context_tokens_max = context_tokens_max;
        self.use_graphs = use_graphs;
        self.kv_format = kv_format;
        self.share_loop_kv = share_loop_kv_req and (context_tokens_max > constants.swa_window_tokens + 64);
        self.schedule = constants.build_visit_schedule();

        tiger.assert_cuda(cuda.squid_cuda_stream_create(&self.stream));
        tiger.assert_cuda(cuda.squid_cuda_stream_create(&self.aux_stream));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&self.fork_event, 1));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&self.join_event, 1));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&self.done_event, 1));

        const expert_groups_bytes: u64 = cuda.squid_sizeof_expert_groups();
        tiger.assert(expert_groups_bytes > 0);
        const has_dspark: bool = (weights.dspark.loaded != 0);

        // Pass 1: compute exact aligned runtime VRAM requirement -> 1 single cudaMalloc!
        const runtime_bytes: u64 = compute_runtime_vram_bytes(
            &self.schedule,
            context_tokens_max,
            kv_format,
            self.share_loop_kv,
            has_dspark,
            expert_groups_bytes,
        );
        self.arena = vram_arena.VRAMArena.init(runtime_bytes);

        // Pass 2: carve every runtime buffer from the single contiguous VRAMArena
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const is_glob: bool = self.schedule[vi].is_global;
            const cap: u32 = visit_kv_cap(&self.schedule, vi, context_tokens_max, self.share_loop_kv);
            const vfmt: constants.KVFormat = if (is_glob) kv_format else .bf16;
            self.kv_cap[vi] = @intCast(cap);
            self.alloc_kv_pair(cap, vfmt, &self.k_cache[vi], &self.v_cache[vi], &self.k_scale[vi], &self.v_scale[vi]);
            self.k_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.kv_dim);
            self.v_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.kv_dim);
            self.attn_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.hidden_dim);
            self.mlp_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.hidden_dim);
        }

        if (has_dspark) {
            var ml: u32 = 0;
            while (ml < constants.mtp_layer_count) : (ml += 1) {
                const is_loc: bool = constants.is_mtp_local_layer(ml);
                const cap: u32 = if (is_loc)
                    (constants.swa_window_tokens + constants.mtp_block_size_max)
                else if (context_tokens_max > 524_288)
                    65_536
                else
                    context_tokens_max;
                const lfmt: constants.KVFormat = if (is_loc) .bf16 else kv_format;
                self.ds_kv_cap[ml] = @intCast(cap);
                self.alloc_kv_pair(cap, lfmt, &self.ds_k_cache[ml], &self.ds_v_cache[ml], &self.ds_k_scale[ml], &self.ds_v_scale[ml]);
            }
            self.ds_f_ring = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.hidden_dim);
        }

        self.split_buf = self.arena.alloc_many(f32, constants.split_buffer_floats);

        const b_max: u64 = constants.mtp_block_size_max;
        self.d_pos = self.arena.alloc_one(i32);
        self.d_tokens = self.arena.alloc_many(i32, b_max);
        self.d_next_tokens = self.arena.alloc_many(i32, b_max);
        self.d_anchor = self.arena.alloc_one(i32);
        self.d_drafts = self.arena.alloc_many(i32, constants.mtp_draft_count_max);
        self.d_draft_conf = self.arena.alloc_many(f32, constants.mtp_draft_count_max);
        self.d_chosen_width = self.arena.alloc_one(i32);
        self.d_accepted = self.arena.alloc_many(i32, b_max);
        self.d_num_acc = self.arena.alloc_one(i32);

        var h_acc_raw: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_acc_raw, @intCast(b_max * @sizeOf(i32)), 0));
        self.h_accepted = @ptrCast(@alignCast(h_acc_raw.?));

        var h_nacc_raw: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_nacc_raw, @sizeOf(i32), 0));
        self.h_num_accepted = @ptrCast(@alignCast(h_nacc_raw.?));

        var h_cw_raw: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_cw_raw, @sizeOf(i32), 0));
        self.h_chosen_width = @ptrCast(@alignCast(h_cw_raw.?));
        var h_ld_raw: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_ld_raw, constants.mtp_draft_count_max * @sizeOf(i32), 0));
        self.h_lookup_drafts = @ptrCast(@alignCast(h_ld_raw.?));

        self.hidden = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.prelude = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.partial = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.blocks = self.arena.alloc_many(f32, b_max * 4 * constants.hidden_dim);
        self.saved = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.prior_last = self.arena.alloc_many(f32, b_max * constants.hidden_dim);

        var s_idx: u32 = 0;
        while (s_idx < 3) : (s_idx += 1) {
            self.attn_in[s_idx] = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
            self.attn_in_h[s_idx] = self.arena.alloc_raw(b_max * constants.hidden_dim * 2);
        }

        self.qrgs = self.arena.alloc_many(f32, b_max * constants.qrgs_out_dim);
        self.k_proj = self.arena.alloc_many(f32, b_max * constants.kv_dim);
        self.v_proj = self.arena.alloc_many(f32, b_max * constants.kv_dim);
        self.attn_gated = self.arena.alloc_many(f32, b_max * constants.query_dim);
        self.attn_gated_h = self.arena.alloc_raw(b_max * constants.query_dim * 2);
        self.o_proj = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.residual = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.mlp_in = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.mlp_normed = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.mlp_normed_h = self.arena.alloc_raw(b_max * constants.hidden_dim * 2);
        self.ctrl_msg = self.arena.alloc_many(f32, 3 * b_max * constants.hidden_dim);
        self.ctrl_cat = self.arena.alloc_many(f32, 3 * b_max * 2 * constants.hidden_dim);
        self.ctrl_d = self.arena.alloc_many(f32, 3 * b_max * constants.gate_rank);
        self.router_logits = self.arena.alloc_many(f32, b_max * constants.expert_count);
        self.route_ids = self.arena.alloc_many(i32, b_max * constants.top_k_experts);
        self.route_w = self.arena.alloc_many(f32, b_max * constants.top_k_experts);
        self.exp_groups = self.arena.alloc_raw(expert_groups_bytes);
        self.moe_act_h = self.arena.alloc_raw(b_max * constants.top_k_experts * constants.intermediate_dim * 2);
        self.moe_down = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.moe_slot_out = self.arena.alloc_many(f32, b_max * constants.top_k_experts * constants.hidden_dim);
        self.taps = self.arena.alloc_many(f32, 4 * b_max * constants.hidden_dim);
        self.ds_mix = self.arena.alloc_many(f32, 3 * b_max * constants.hidden_dim);
        self.ds_kv = self.arena.alloc_many(f32, b_max * 2 * constants.kv_dim);
        self.ds_ws = self.arena.alloc_many(f32, constants.dspark_workspace_floats);
        self.lm_blk_val = self.arena.alloc_many(f32, b_max * 1024);
        self.lm_blk_idx = self.arena.alloc_many(i32, b_max * 1024);
        self.lm_logits = self.arena.alloc_many(f32, b_max * constants.vocab_unpadded_count);
        self.d_draft_probs = self.arena.alloc_many(f32, constants.mtp_draft_count_max * constants.vocab_unpadded_count);
        self.d_draft_q_sel = self.arena.alloc_many(f32, constants.mtp_draft_count_max);
        self.d_draft_is_delta = self.arena.alloc_one(i32);
        self.d_sample_cfg = self.arena.alloc_one(cuda.CSampleConfig);
        const zero_cfg: cuda.CSampleConfig = .{};
        const zero_i32: i32 = 0;
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_sample_cfg), @ptrCast(&zero_cfg), @sizeOf(cuda.CSampleConfig), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_draft_is_delta), @ptrCast(&zero_i32), @sizeOf(i32), self.stream));

        // Fast batched prefill arena (B <= 256)
        const pb: u64 = prefill_batch_max;
        self.p_tokens = self.arena.alloc_many(i32, pb);
        self.p_pos_vec = self.arena.alloc_many(i32, prefill_sub_chunks_max);
        self.p_hidden = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_prelude = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_partial = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_blocks = self.arena.alloc_many(f32, pb * 4 * constants.hidden_dim);
        self.p_saved = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_prior_last = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        s_idx = 0;
        while (s_idx < 3) : (s_idx += 1) {
            self.p_attn_in[s_idx] = self.arena.alloc_many(f32, pb * constants.hidden_dim);
            self.p_attn_in_h[s_idx] = self.arena.alloc_raw(pb * constants.hidden_dim * 2);
        }
        self.p_qrgs = self.arena.alloc_many(f32, pb * constants.qrgs_out_dim);
        self.p_k_proj = self.arena.alloc_many(f32, pb * constants.kv_dim);
        self.p_v_proj = self.arena.alloc_many(f32, pb * constants.kv_dim);
        self.p_attn_gated_h = self.arena.alloc_raw(pb * constants.query_dim * 2);
        self.p_o_proj = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_residual = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_mlp_in = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_mlp_normed = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_mlp_normed_h = self.arena.alloc_raw(pb * constants.hidden_dim * 2);
        self.p_router_logits = self.arena.alloc_many(f32, pb * constants.expert_count);
        self.p_route_ids = self.arena.alloc_many(i32, pb * constants.top_k_experts);
        self.p_route_w = self.arena.alloc_many(f32, pb * constants.top_k_experts);
        self.p_exp_offsets = self.arena.alloc_many(i32, 1024);
        self.p_sorted_tok = self.arena.alloc_many(i32, pb * constants.top_k_experts);
        self.p_sorted_slot = self.arena.alloc_many(i32, pb * constants.top_k_experts);
        self.p_sorted_w = self.arena.alloc_many(f32, pb * constants.top_k_experts);
        self.p_moe_act_h = self.arena.alloc_raw(pb * constants.top_k_experts * constants.intermediate_dim * 2);
        self.p_moe_slot_out = self.arena.alloc_many(f32, pb * constants.top_k_experts * constants.hidden_dim);
        self.p_taps = self.arena.alloc_many(f32, 4 * pb * constants.hidden_dim);
        self.p_w_f16 = self.arena.alloc_raw(@as(u64, constants.qrgs_out_dim) * constants.hidden_dim * 2);
        self.p_ctrl_msg = self.arena.alloc_many(f32, pb * constants.hidden_dim);
        self.p_ctrl_cat_h = self.arena.alloc_raw(pb * 2 * constants.hidden_dim * 2);
        self.p_ctrl_d = self.arena.alloc_many(f32, pb * constants.gate_rank);
        self.p_ctrl_d_h = self.arena.alloc_raw(pb * constants.gate_rank * 2);
        // Alias p_ctrl_f and p_ctrl_g inside p_qrgs (non-overlapping lifetimes: 2*2304=4608 <= 6552)
        self.p_ctrl_f = self.p_qrgs;
        self.p_ctrl_g = self.p_qrgs + pb * constants.hidden_dim;

        // Paired TigerStyle invariant: Pass 1 == Pass 2!
        self.arena.assert_exact_fill();

        // Pre-initialize cuBLAS handle during startup so Chunk 0 prefill pays 0ms cold handle creation.
        cuda.squid_launch_prefill_warmup_cublas(self.stream);

        const rt_gib: f64 = @as(f64, @floatFromInt(self.arena.capacity_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[engine] Initialized max_ctx={d} | kv={s}{s} | runtime VRAM={d:.2} GiB (1 arena, {d} slices) | graphs={s}\n",
            .{ context_tokens_max, kv_format.name(), if (self.share_loop_kv) " (share-loop-kv)" else "", rt_gib, self.arena.allocation_count, if (use_graphs) "on" else "off" },
        );
        self.reset();
    }

    pub fn deinit(self: *Engine) void {
        if (self.b1_graph_exec) |g| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(g));
        for (&self.spec_graph_exec) |*g| {
            if (g.*) |exec| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(exec));
            g.* = null;
        }
        if (self.draft_graph_exec) |g| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(g));
        if (self.verify_b8_graph_exec) |g| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(g));
        if (self.adaptive_graph) |g| tiger.assert_cuda(cuda.squid_adaptive_graph_destroy(g));
        for (&self.prefill_b8_graph_exec) |*g| {
            if (g.*) |exec| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(exec));
            g.* = null;
        }
        if (self.h_accepted) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        if (self.h_num_accepted) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        if (self.h_chosen_width) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        if (self.h_lookup_drafts) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        self.deinit_prefix_cache();
        self.arena.deinit();
        if (self.done_event) |e| tiger.assert_cuda(cuda.squid_cuda_event_destroy(e));
        if (self.fork_event) |e| tiger.assert_cuda(cuda.squid_cuda_event_destroy(e));
        if (self.join_event) |e| tiger.assert_cuda(cuda.squid_cuda_event_destroy(e));
        if (self.aux_stream) |s| tiger.assert_cuda(cuda.squid_cuda_stream_destroy(s));
        if (self.stream) |s| tiger.assert_cuda(cuda.squid_cuda_stream_destroy(s));
        self.* = .{};
    }

    pub fn reset(self: *Engine) void {
        tiger.assert(self.stream != null);
        self.current_position = 0;
        tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.d_pos), 0, @sizeOf(i32), self.stream));
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.k_ring[vi]), 0, constants.sconv_ring_slots * constants.kv_dim * 4, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.v_ring[vi]), 0, constants.sconv_ring_slots * constants.kv_dim * 4, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.attn_ring[vi]), 0, constants.sconv_ring_slots * constants.hidden_dim * 4, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.mlp_ring[vi]), 0, constants.sconv_ring_slots * constants.hidden_dim * 4, self.stream));
        }
        if (self.ds_f_ring) |fr| {
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(fr), 0, constants.sconv_ring_slots * constants.hidden_dim * 4, self.stream));
        }
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
    }

    /// Allocate static host-RAM prefix KV cache (zero extra VRAM, zero malloc per request).
    /// Stores 4 local-state+ring checkpoints (~193 MiB each) + a 4K-token global-KV side-request shield.
    pub fn enable_prefix_cache(self: *Engine) void {
        tiger.assert(self.weights != null);
        if (self.prefix_cache.enabled) return;

        const local_bytes: usize = self.compute_local_state_bytes();
        const shield_bytes: usize = self.compute_global_kv_prefix_bytes(side_shield_max_tokens);
        tiger.assert(local_bytes > 0 and shield_bytes > 0);

        self.prefix_cache.local_state_bytes = local_bytes;
        self.prefix_cache.session_tokens = tiger.alloc_host_zeroed(i32, self.context_tokens_max);
        for (&self.prefix_cache.checkpoints) |*ckpt| {
            ckpt.data = tiger.alloc_host_zeroed(u8, local_bytes);
            ckpt.pos = 0;
            ckpt.valid = false;
        }
        self.prefix_cache.side_shield_buf = tiger.alloc_host_zeroed(u8, shield_bytes);
        self.prefix_cache.session_len = 0;
        self.prefix_cache.gpu_live_at_pos = -1;
        self.prefix_cache.enabled = true;

        const host_mib: f64 = @as(f64, @floatFromInt(local_bytes * prefix_checkpoint_count + shield_bytes + self.context_tokens_max * 4)) / (1024.0 * 1024.0);
        tiger.print_stdout(
            "[cache] Prefix KV cache enabled ({d} checkpoints + {d}-tok side-request shield, host RAM={d:.1} MiB, extra VRAM=0.0 MiB)\n",
            .{ prefix_checkpoint_count, side_shield_max_tokens, host_mib },
        );
    }

    fn deinit_prefix_cache(self: *Engine) void {
        if (!self.prefix_cache.enabled) return;
        if (self.prefix_cache.session_tokens.len > 0) tiger.free_host_slice(i32, self.prefix_cache.session_tokens);
        for (&self.prefix_cache.checkpoints) |*ckpt| {
            if (ckpt.data.len > 0) tiger.free_host_slice(u8, ckpt.data);
        }
        if (self.prefix_cache.side_shield_buf.len > 0) tiger.free_host_slice(u8, self.prefix_cache.side_shield_buf);
        self.prefix_cache = .{};
    }

    fn compute_local_state_bytes(self: *const Engine) usize {
        var total: usize = 0;
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const is_shared_tail = self.share_loop_kv and is_shared_loop_global_visit(vi);
            if (!self.schedule[vi].is_global or is_shared_tail) {
                const vfmt: constants.KVFormat = if (is_shared_tail) self.kv_format else .bf16;
                const sz = global_kv_bytes_for_tokens(vfmt, @intCast(self.kv_cap[vi]));
                total += 2 * sz.data + 2 * sz.scale;
            }
            total += 2 * constants.sconv_ring_slots * constants.kv_dim * 4;
            total += 2 * constants.sconv_ring_slots * constants.hidden_dim * 4;
        }
        if (self.ds_f_ring != null) {
            var ml: u32 = 0;
            while (ml < constants.mtp_layer_count) : (ml += 1) {
                if (constants.is_mtp_local_layer(ml)) {
                    total += 2 * @as(usize, @intCast(self.ds_kv_cap[ml])) * constants.kv_dim * 2;
                }
            }
            total += constants.sconv_ring_slots * constants.hidden_dim * 4;
        }
        return total;
    }

    fn global_kv_bytes_for_tokens(kv_format: constants.KVFormat, token_count: u32) struct { data: usize, scale: usize } {
        const tc: usize = token_count;
        return switch (kv_format) {
            .bf16 => .{ .data = tc * constants.kv_dim * 2, .scale = 0 },
            .fp8 => .{ .data = tc * constants.kv_dim, .scale = tc * constants.kv_head_count * 2 },
            .q4 => .{ .data = tc * (constants.kv_dim / 2), .scale = tc * constants.kv_q4_scales_per_token * 2 },
        };
    }

    fn compute_global_kv_prefix_bytes(self: *const Engine, token_count: u32) usize {
        const sz = global_kv_bytes_for_tokens(self.kv_format, token_count);
        var total: usize = 0;
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            if (self.schedule[vi].is_global and !(self.share_loop_kv and is_shared_loop_global_visit(vi))) {
                total += 2 * sz.data + 2 * sz.scale;
            }
        }
        if (self.ds_f_ring != null) {
            total += 2 * sz.data + 2 * sz.scale; // DSpark layer 1 (global)
        }
        return total;
    }

    fn save_local_state_checkpoint(self: *Engine, slot_idx: usize, pos: u32) void {
        tiger.assert(self.prefix_cache.enabled and slot_idx < prefix_checkpoint_count);
        const ckpt = &self.prefix_cache.checkpoints[slot_idx];
        const dst = ckpt.data;
        var off: usize = 0;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const is_shared_tail = self.share_loop_kv and is_shared_loop_global_visit(vi);
            if (!self.schedule[vi].is_global or is_shared_tail) {
                const vfmt: constants.KVFormat = if (is_shared_tail) self.kv_format else .bf16;
                const sz = global_kv_bytes_for_tokens(vfmt, @intCast(self.kv_cap[vi]));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), self.k_cache[vi].?, sz.data, self.stream));
                off += sz.data;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), self.v_cache[vi].?, sz.data, self.stream));
                off += sz.data;
                if (sz.scale > 0) {
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), self.k_scale[vi].?, sz.scale, self.stream));
                    off += sz.scale;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), self.v_scale[vi].?, sz.scale, self.stream));
                    off += sz.scale;
                }
            }
            const kv_ring_bytes: usize = constants.sconv_ring_slots * constants.kv_dim * 4;
            const h_ring_bytes: usize = constants.sconv_ring_slots * constants.hidden_dim * 4;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), @ptrCast(self.k_ring[vi]), kv_ring_bytes, self.stream));
            off += kv_ring_bytes;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), @ptrCast(self.v_ring[vi]), kv_ring_bytes, self.stream));
            off += kv_ring_bytes;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), @ptrCast(self.attn_ring[vi]), h_ring_bytes, self.stream));
            off += h_ring_bytes;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), @ptrCast(self.mlp_ring[vi]), h_ring_bytes, self.stream));
            off += h_ring_bytes;
        }
        if (self.ds_f_ring) |fr| {
            var ml: u32 = 0;
            while (ml < constants.mtp_layer_count) : (ml += 1) {
                if (constants.is_mtp_local_layer(ml)) {
                    const kv_bytes: usize = @as(usize, @intCast(self.ds_kv_cap[ml])) * constants.kv_dim * 2;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), self.ds_k_cache[ml].?, kv_bytes, self.stream));
                    off += kv_bytes;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), self.ds_v_cache[ml].?, kv_bytes, self.stream));
                    off += kv_bytes;
                }
            }
            const fr_bytes: usize = constants.sconv_ring_slots * constants.hidden_dim * 4;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(dst[off..].ptr), @ptrCast(fr), fr_bytes, self.stream));
            off += fr_bytes;
        }
        tiger.assert(off == self.prefix_cache.local_state_bytes);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
        ckpt.pos = pos;
        ckpt.valid = true;
    }

    fn restore_local_state_checkpoint(self: *Engine, slot_idx: usize) void {
        tiger.assert(self.prefix_cache.enabled and slot_idx < prefix_checkpoint_count);
        const ckpt = &self.prefix_cache.checkpoints[slot_idx];
        tiger.assert(ckpt.valid);
        const src = ckpt.data;
        var off: usize = 0;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const is_shared_tail = self.share_loop_kv and is_shared_loop_global_visit(vi);
            if (!self.schedule[vi].is_global or is_shared_tail) {
                const vfmt: constants.KVFormat = if (is_shared_tail) self.kv_format else .bf16;
                const sz = global_kv_bytes_for_tokens(vfmt, @intCast(self.kv_cap[vi]));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.k_cache[vi].?, @ptrCast(src[off..].ptr), sz.data, self.stream));
                off += sz.data;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.v_cache[vi].?, @ptrCast(src[off..].ptr), sz.data, self.stream));
                off += sz.data;
                if (sz.scale > 0) {
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.k_scale[vi].?, @ptrCast(src[off..].ptr), sz.scale, self.stream));
                    off += sz.scale;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.v_scale[vi].?, @ptrCast(src[off..].ptr), sz.scale, self.stream));
                    off += sz.scale;
                }
            }
            const kv_ring_bytes: usize = constants.sconv_ring_slots * constants.kv_dim * 4;
            const h_ring_bytes: usize = constants.sconv_ring_slots * constants.hidden_dim * 4;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.k_ring[vi]), @ptrCast(src[off..].ptr), kv_ring_bytes, self.stream));
            off += kv_ring_bytes;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.v_ring[vi]), @ptrCast(src[off..].ptr), kv_ring_bytes, self.stream));
            off += kv_ring_bytes;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.attn_ring[vi]), @ptrCast(src[off..].ptr), h_ring_bytes, self.stream));
            off += h_ring_bytes;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.mlp_ring[vi]), @ptrCast(src[off..].ptr), h_ring_bytes, self.stream));
            off += h_ring_bytes;
        }
        if (self.ds_f_ring) |fr| {
            var ml: u32 = 0;
            while (ml < constants.mtp_layer_count) : (ml += 1) {
                if (constants.is_mtp_local_layer(ml)) {
                    const kv_bytes: usize = @as(usize, @intCast(self.ds_kv_cap[ml])) * constants.kv_dim * 2;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.ds_k_cache[ml].?, @ptrCast(src[off..].ptr), kv_bytes, self.stream));
                    off += kv_bytes;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.ds_v_cache[ml].?, @ptrCast(src[off..].ptr), kv_bytes, self.stream));
                    off += kv_bytes;
                }
            }
            const fr_bytes: usize = constants.sconv_ring_slots * constants.hidden_dim * 4;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(fr), @ptrCast(src[off..].ptr), fr_bytes, self.stream));
            off += fr_bytes;
        }
        tiger.assert(off == self.prefix_cache.local_state_bytes);
    }

    fn copy_global_kv_shield(self: *Engine, token_count: u32, save_d2h: bool) void {
        tiger.assert(self.prefix_cache.enabled and token_count > 0 and token_count <= side_shield_max_tokens);
        const buf = self.prefix_cache.side_shield_buf;
        const sz = global_kv_bytes_for_tokens(self.kv_format, token_count);
        var off: usize = 0;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            if (!self.schedule[vi].is_global or (self.share_loop_kv and is_shared_loop_global_visit(vi))) continue;
            if (save_d2h) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.k_cache[vi].?, sz.data, self.stream));
                off += sz.data;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.v_cache[vi].?, sz.data, self.stream));
                off += sz.data;
                if (sz.scale > 0) {
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.k_scale[vi].?, sz.scale, self.stream));
                    off += sz.scale;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.v_scale[vi].?, sz.scale, self.stream));
                    off += sz.scale;
                }
            } else {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.k_cache[vi].?, @ptrCast(buf[off..].ptr), sz.data, self.stream));
                off += sz.data;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.v_cache[vi].?, @ptrCast(buf[off..].ptr), sz.data, self.stream));
                off += sz.data;
                if (sz.scale > 0) {
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.k_scale[vi].?, @ptrCast(buf[off..].ptr), sz.scale, self.stream));
                    off += sz.scale;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.v_scale[vi].?, @ptrCast(buf[off..].ptr), sz.scale, self.stream));
                    off += sz.scale;
                }
            }
        }
        if (self.ds_f_ring != null) {
            const ds_tokens: u32 = @min(token_count, @as(u32, @intCast(self.ds_kv_cap[1])));
            const dsz = global_kv_bytes_for_tokens(self.kv_format, ds_tokens);
            if (save_d2h) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.ds_k_cache[1].?, dsz.data, self.stream));
                off += dsz.data;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.ds_v_cache[1].?, dsz.data, self.stream));
                off += dsz.data;
                if (dsz.scale > 0) {
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.ds_k_scale[1].?, dsz.scale, self.stream));
                    off += dsz.scale;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(buf[off..].ptr), self.ds_v_scale[1].?, dsz.scale, self.stream));
                    off += dsz.scale;
                }
            } else {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.ds_k_cache[1].?, @ptrCast(buf[off..].ptr), dsz.data, self.stream));
                off += dsz.data;
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.ds_v_cache[1].?, @ptrCast(buf[off..].ptr), dsz.data, self.stream));
                off += dsz.data;
                if (dsz.scale > 0) {
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.ds_k_scale[1].?, @ptrCast(buf[off..].ptr), dsz.scale, self.stream));
                    off += dsz.scale;
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(self.ds_v_scale[1].?, @ptrCast(buf[off..].ptr), dsz.scale, self.stream));
                    off += dsz.scale;
                }
            }
        }
        tiger.assert(off <= buf.len);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
    }

    /// Generate up to `max_new_tokens` into `out_tokens` (and optionally invoke `on_token`).
    /// ZERO heap allocations occur during the entire prefill + decode execution!
    pub fn generate(
        self: *Engine,
        prompt: []const i32,
        max_new_tokens_in: u32,
        num_drafts_in: u32,
        out_tokens: []i32,
        on_token: ?TokenCallbackFn,
        callback_context: ?*anyopaque,
    ) GenerateStats {
        return self.generate_with_hint(prompt, 0, max_new_tokens_in, num_drafts_in, out_tokens, on_token, callback_context);
    }

    pub fn generate_with_hint(
        self: *Engine,
        prompt: []const i32,
        system_prefix_hint: u32,
        max_new_tokens_in: u32,
        num_drafts_in: u32,
        out_tokens: []i32,
        on_token: ?TokenCallbackFn,
        callback_context: ?*anyopaque,
    ) GenerateStats {
        tiger.assert(self.weights != null);
        var stats: GenerateStats = .{ .prompt_tokens = @intCast(prompt.len) };
        if (prompt.len == 0 or max_new_tokens_in == 0) return stats;
        if (prompt.len >= self.context_tokens_max) return stats;

        var max_new_tokens: u32 = max_new_tokens_in;
        const room: u32 = self.context_tokens_max - @as(u32, @intCast(prompt.len));
        if (max_new_tokens > room) max_new_tokens = room;
        if (out_tokens.len > 0 and max_new_tokens > out_tokens.len) {
            max_new_tokens = @intCast(out_tokens.len);
        }

        const w = self.weights.?;
        const num_drafts: u32 = if (num_drafts_in > constants.mtp_draft_count_max) constants.mtp_draft_count_max else num_drafts_in;
        const use_mtp: bool = (num_drafts > 0 and w.dspark.loaded != 0);
        const prompt_len: u32 = @intCast(prompt.len);

        // Upload per-request sampling config to device (zero CUDA graph rebuild!).
        const cfg_h: cuda.CSampleConfig = .{
            .inv_temp = if (self.sample_params.temperature >= 1e-4) (1.0 / self.sample_params.temperature) else 0.0,
            .top_p = self.sample_params.top_p,
            .top_k = self.sample_params.top_k,
            .pad = 0,
            .seed = self.sample_params.seed,
        };
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_sample_cfg), @ptrCast(&cfg_h), @sizeOf(cuda.CSampleConfig), self.stream));

        // --- PREFIX KV CACHE MATCHING & SIDE-REQUEST SHIELDING ---
        const prefill_start_ns: u64 = tiger.clock_monotonic_ns();
        var start_pos: u32 = 0;
        var shield_tokens: u32 = 0;

        if (self.prefix_cache.enabled) {
            const max_match: u32 = @min(self.prefix_cache.session_len, if (prompt_len > 1) prompt_len - 1 else 0);
            var common: u32 = 0;
            while (common < max_match and self.prefix_cache.session_tokens[common] == prompt[common]) : (common += 1) {}

            // Short non-matching side request while holding a long session? Shield global KV prefix!
            const side_need: u32 = prompt_len + max_new_tokens + 16;
            if (self.prefix_cache.session_len >= 1024 and common < 512 and side_need <= side_shield_max_tokens) {
                shield_tokens = @min(self.prefix_cache.session_len, side_need);
                self.copy_global_kv_shield(shield_tokens, true);
                self.reset();
            } else if (common > 0 and use_mtp == self.prefix_cache.session_use_mtp) {
                // Target global KV caches hold full `context_tokens_max` (up to 1M) without wrapping.
                // DSpark's global KV ring (64K at 1M ctx) only proposes draft tokens that are strictly
                // verified by the target model, so never clamp target prefix reuse at 65,536!
                const reusable_limit: u32 = common;

                // Check 1: exact live GPU match (0 ms restore!)
                var matched_live: bool = false;
                if (self.prefix_cache.gpu_live_at_pos > 0 and @as(u32, @intCast(self.prefix_cache.gpu_live_at_pos)) <= reusable_limit) {
                    start_pos = @intCast(self.prefix_cache.gpu_live_at_pos);
                    matched_live = true;
                }
                // Check 2: best host-RAM checkpoint (may be strictly further than live, e.g. prefill_end or system_prefix!)
                var best_slot: ?usize = null;
                for (self.prefix_cache.checkpoints, 0..) |ckpt, sidx| {
                    if (ckpt.valid and ckpt.pos <= reusable_limit and ckpt.pos > start_pos) {
                        start_pos = ckpt.pos;
                        best_slot = sidx;
                        matched_live = false;
                    }
                }

                if (start_pos > 0) {
                    if (!matched_live) {
                        self.restore_local_state_checkpoint(best_slot.?);
                    }
                    self.current_position = @intCast(start_pos);
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
                    stats.cached_tokens = start_pos;
                } else {
                    self.reset();
                    for (&self.prefix_cache.checkpoints) |*ckpt| ckpt.valid = false;
                }
            } else {
                self.reset();
                for (&self.prefix_cache.checkpoints) |*ckpt| ckpt.valid = false;
            }
        } else {
            self.reset();
        }
        defer if (shield_tokens > 0) {
            self.copy_global_kv_shield(shield_tokens, false);
            self.prefix_cache.gpu_live_at_pos = -1;
        };

        // --- PREFILL (ONLY prompt[start_pos .. prompt_len]) ---
        // Invalidate any stale checkpoints past the branch point `start_pos` BEFORE prefill writes new KV!
        if (self.prefix_cache.enabled and shield_tokens == 0) {
            for (&self.prefix_cache.checkpoints) |*ckpt| {
                if (ckpt.valid and ckpt.pos > start_pos) ckpt.valid = false;
            }
        }
        const rem_tokens: u32 = prompt_len - start_pos;
        if (self.use_graphs and rem_tokens > constants.mtp_block_size_max) {
            self.ensure_prefill_b8_graph(use_mtp);
        }

        // Intermediate checkpoint targets (only when updating main session, not shielded side requests)
        var target_sys: u32 = 0;
        var target_2k: u32 = 0;
        if (self.prefix_cache.enabled and shield_tokens == 0) {
            if (system_prefix_hint >= 128 and system_prefix_hint > start_pos and system_prefix_hint < prompt_len) {
                target_sys = system_prefix_hint;
            }
            const p2k: u32 = ((prompt_len - 1) / 2048) * 2048;
            if (p2k >= 2048 and p2k > start_pos and p2k < prompt_len and p2k != target_sys) {
                target_2k = p2k;
            }
        }

        // Run prefill segment helper up to `seg_end` (strictly < prompt_len for intermediate checkpoints)
        var i: u32 = start_pos;
        const targets = [_]struct { pos: u32, slot: usize }{
            if (target_sys > 0 and (target_2k == 0 or target_sys < target_2k)) .{ .pos = target_sys, .slot = 0 } else .{ .pos = target_2k, .slot = 1 },
            if (target_sys > 0 and target_2k > 0) (if (target_sys < target_2k) .{ .pos = target_2k, .slot = 1 } else .{ .pos = target_sys, .slot = 0 }) else .{ .pos = 0, .slot = 0 },
        };
        for (targets) |tgt| {
            if (tgt.pos > i and tgt.pos < prompt_len) {
                i = self.prefill_range(prompt, i, tgt.pos, use_mtp, false);
                self.save_local_state_checkpoint(tgt.slot, i);
            }
        }

        // Final prefill segment up to prompt_len (computing lm_head on the final chunk)
        const last_batch: u32 = self.prefill_final_range(prompt, i, prompt_len, use_mtp);

        var first_tok: i32 = 0;
        tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(&first_tok), @ptrCast(self.d_next_tokens + (last_batch - 1)), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        // Update session tokens & save prefill_end checkpoint (Slot 2) BEFORE decode overwrites rings/local KV!
        if (self.prefix_cache.enabled and shield_tokens == 0) {
            @memcpy(self.prefix_cache.session_tokens[start_pos..prompt_len], prompt[start_pos..prompt_len]);
            self.prefix_cache.session_len = prompt_len;
            self.prefix_cache.session_use_mtp = use_mtp;
            self.prefix_cache.gpu_live_at_pos = @intCast(prompt_len);
            for (&self.prefix_cache.checkpoints) |*ckpt| {
                if (ckpt.valid and ckpt.pos > prompt_len) ckpt.valid = false;
            }
            if (prompt_len >= 128) {
                self.save_local_state_checkpoint(2, prompt_len);
            }
        }
        stats.prefill_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - prefill_start_ns)) / 1_000_000.0;

        if (out_tokens.len > 0) out_tokens[0] = first_tok;
        stats.generated_tokens = 1;
        stats.last_token_id = first_tok;
        if (on_token) |cb| {
            if (!cb(callback_context.?, first_tok)) return stats;
        }
        if (first_tok == constants.eos_token_id or stats.generated_tokens >= max_new_tokens) return stats;

        // Prepare decode CUDA Graph before starting decode timer
        const use_adaptive: bool = use_mtp and self.use_graphs and num_drafts == constants.mtp_draft_count_max and self.spec_threshold > 0.0;
        if (!use_mtp) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(&first_tok), @sizeOf(i32), self.stream));
            self.ensure_b1_graph();
        } else if (use_adaptive) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_anchor), @ptrCast(&first_tok), @sizeOf(i32), self.stream));
            self.ensure_adaptive_graphs();
        } else {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_anchor), @ptrCast(&first_tok), @sizeOf(i32), self.stream));
            self.ensure_spec_graph(num_drafts);
        }
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        // --- DECODE ---
        var prev_fed_tok: i32 = first_tok;
        var decode_stop_now: bool = false;
        const decode_start_ns: u64 = tiger.clock_monotonic_ns();
        if (!use_mtp) {
            while (stats.generated_tokens < max_new_tokens and @as(u32, @intCast(self.current_position)) < self.context_tokens_max) {
                if (self.use_graphs) {
                    tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.b1_graph_exec.?, self.stream));
                } else {
                    self.record_target_forward(1, false, true, false);
                    cuda.squid_launch_inc_pos_and_copy_tok(self.d_pos, &self.d_tokens[0], &self.d_next_tokens[0], self.stream);
                }
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_accepted.?), @ptrCast(self.d_next_tokens), @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

                if (self.prefix_cache.enabled and shield_tokens == 0 and self.prefix_cache.session_len < self.context_tokens_max) {
                    self.prefix_cache.session_tokens[self.prefix_cache.session_len] = prev_fed_tok;
                    self.prefix_cache.session_len += 1;
                }
                const tok: i32 = self.h_accepted.?[0];
                prev_fed_tok = tok;
                if (out_tokens.len > 0) out_tokens[stats.generated_tokens] = tok;
                stats.generated_tokens += 1;
                stats.last_token_id = tok;
                self.current_position += 1;
                if (on_token) |cb| {
                    if (!cb(callback_context.?, tok)) {
                        decode_stop_now = true;
                        break;
                    }
                }
                if (tok == constants.eos_token_id) {
                    decode_stop_now = true;
                    break;
                }
            }
        } else if (use_adaptive) {
            // Hybrid O(1) Prompt-Lookup + Overlapped Adaptive {2, 4, 8} DSpark Speculative Decode!
            // Seed prompt-lookup ring with trailing prompt tokens + first_tok (<=8192 tok -> <25 us CPU).
            if (self.use_prompt_lookup) {
                self.prompt_lookup.reset();
                const seed_start: usize = if (prompt.len > PromptLookupIndex.hist_cap) (prompt.len - PromptLookupIndex.hist_cap) else 0;
                for (prompt[seed_start..]) |ptok| self.prompt_lookup.push(ptok);
                self.prompt_lookup.push(first_tok);
            }

            var draft_in_flight: bool = false;
            while (stats.generated_tokens < max_new_tokens and @as(u32, @intCast(self.current_position)) + constants.mtp_block_size_max <= self.context_tokens_max) {
                var lookup_hit: bool = false;
                if (self.use_prompt_lookup) {
                    const out_ptr: *[constants.mtp_draft_count_max]i32 = @ptrCast(self.h_lookup_drafts.?);
                    if (self.prompt_lookup.match(out_ptr)) {
                        lookup_hit = true;
                        const one_i32: i32 = 1;
                        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                            @ptrCast(self.d_draft_is_delta),
                            @ptrCast(&one_i32),
                            @sizeOf(i32),
                            self.stream,
                        ));
                        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                            @ptrCast(self.d_drafts),
                            @ptrCast(self.h_lookup_drafts.?),
                            constants.mtp_draft_count_max * @sizeOf(i32),
                            self.stream,
                        ));
                        tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.verify_b8_graph_exec.?, self.stream));
                        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                        draft_in_flight = false;
                        stats.lookup_steps += 1;
                    }
                }
                if (!lookup_hit) {
                    if (!draft_in_flight) {
                        tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.draft_graph_exec.?, self.stream));
                    }
                    tiger.assert_cuda(cuda.squid_adaptive_graph_launch(self.adaptive_graph.?, self.stream));
                    tiger.assert_cuda(cuda.squid_cuda_event_record(self.done_event, self.stream));
                    // Overlap next round's DSpark draft while host reads verified tokens & emits SSE!
                    tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.draft_graph_exec.?, self.stream));
                    draft_in_flight = true;
                    tiger.assert_cuda(cuda.squid_cuda_event_sync(self.done_event));
                }

                const n_acc_i32: i32 = self.h_num_accepted.?.*;
                const chosen_w_i32: i32 = if (lookup_hit) constants.mtp_block_size_max else self.h_chosen_width.?.*;
                tiger.assert(chosen_w_i32 == 2 or chosen_w_i32 == 4 or chosen_w_i32 == 8);
                tiger.assert(n_acc_i32 >= 1 and n_acc_i32 <= chosen_w_i32);
                const n_acc: u32 = @intCast(n_acc_i32);

                if (self.prefix_cache.enabled and shield_tokens == 0) {
                    var ci: u32 = 0;
                    while (ci < n_acc and self.prefix_cache.session_len < self.context_tokens_max) : (ci += 1) {
                        const fed: i32 = if (ci == 0) prev_fed_tok else self.h_accepted.?[ci - 1];
                        self.prefix_cache.session_tokens[self.prefix_cache.session_len] = fed;
                        self.prefix_cache.session_len += 1;
                    }
                }
                prev_fed_tok = self.h_accepted.?[n_acc - 1];

                stats.spec_steps += 1;
                stats.total_drafts += @intCast(chosen_w_i32 - 1);
                stats.total_accepted += (n_acc - 1);
                self.current_position += n_acc_i32;

                var idx: u32 = 0;
                while (idx < n_acc and stats.generated_tokens < max_new_tokens) : (idx += 1) {
                    const tok: i32 = self.h_accepted.?[idx];
                    if (self.use_prompt_lookup) self.prompt_lookup.push(tok);
                    if (out_tokens.len > 0) out_tokens[stats.generated_tokens] = tok;
                    stats.generated_tokens += 1;
                    stats.last_token_id = tok;
                    if (on_token) |cb| {
                        if (!cb(callback_context.?, tok)) {
                            decode_stop_now = true;
                            break;
                        }
                    }
                    if (tok == constants.eos_token_id) {
                        decode_stop_now = true;
                        break;
                    }
                }
                if (decode_stop_now) break;
            }
            // Drain any trailing overlapped draft cleanly before returning or falling back to B=1 tail
            if (draft_in_flight) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
            }
        } else {
            const batch: i32 = @intCast(num_drafts + 1);
            while (stats.generated_tokens < max_new_tokens and @as(u32, @intCast(self.current_position)) + @as(u32, @intCast(batch)) <= self.context_tokens_max) {
                if (self.use_graphs) {
                    tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.spec_graph_exec[num_drafts].?, self.stream));
                } else {
                    const ds_ks_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_k_scale[0], self.ds_k_scale[1], self.ds_k_scale[2] };
                    const ds_vs_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_v_scale[0], self.ds_v_scale[1], self.ds_v_scale[2] };
                    cuda.squid_launch_dspark_draft(
                        w.embed.?,
                        w.embed_norm.?,
                        &w.dspark,
                        self.d_anchor,
                        self.d_pos,
                        @intCast(num_drafts),
                        &self.ds_k_cache,
                        &self.ds_v_cache,
                        &ds_ks_const,
                        &ds_vs_const,
                        &self.ds_kv_cap,
                        @intFromEnum(self.kv_format),
                        self.split_buf,
                        self.ds_f_ring.?,
                        self.ds_ws,
                        self.d_drafts,
                        null,
                        self.d_sample_cfg,
                        self.d_draft_probs,
                        self.d_draft_q_sel,
                        self.d_draft_is_delta,
                        self.stream,
                    );
                    cuda.squid_launch_assemble_verify_tokens(self.d_anchor, self.d_drafts, self.d_tokens, @intCast(num_drafts), self.stream);
                    self.record_target_forward(batch, true, true, true);
                    cuda.squid_launch_spec_verify(self.d_next_tokens, self.d_drafts, batch, self.d_accepted, self.d_num_acc, self.d_pos, self.d_anchor, self.stream);
                }

                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_num_accepted.?), @ptrCast(self.d_num_acc), @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_accepted.?), @ptrCast(self.d_accepted), @as(usize, @intCast(batch)) * @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

                const n_acc_i32: i32 = self.h_num_accepted.?.*;
                tiger.assert(n_acc_i32 >= 1 and n_acc_i32 <= batch);
                const n_acc: u32 = @intCast(n_acc_i32);

                // The n_acc tokens that were actually fed into the KV cache are [prev_fed_tok, h_accepted[0..n_acc-1]]!
                if (self.prefix_cache.enabled and shield_tokens == 0) {
                    var ci: u32 = 0;
                    while (ci < n_acc and self.prefix_cache.session_len < self.context_tokens_max) : (ci += 1) {
                        const fed: i32 = if (ci == 0) prev_fed_tok else self.h_accepted.?[ci - 1];
                        self.prefix_cache.session_tokens[self.prefix_cache.session_len] = fed;
                        self.prefix_cache.session_len += 1;
                    }
                }
                prev_fed_tok = self.h_accepted.?[n_acc - 1];

                stats.spec_steps += 1;
                stats.total_drafts += num_drafts;
                stats.total_accepted += (n_acc - 1);
                self.current_position += n_acc_i32;

                var idx: u32 = 0;
                while (idx < n_acc and stats.generated_tokens < max_new_tokens) : (idx += 1) {
                    const tok: i32 = self.h_accepted.?[idx];
                    if (out_tokens.len > 0) out_tokens[stats.generated_tokens] = tok;
                    stats.generated_tokens += 1;
                    stats.last_token_id = tok;
                    if (on_token) |cb| {
                        if (!cb(callback_context.?, tok)) {
                            decode_stop_now = true;
                            break;
                        }
                    }
                    if (tok == constants.eos_token_id) {
                        decode_stop_now = true;
                        break;
                    }
                }
                if (decode_stop_now) break;
            }
        }
        // Safe B=1 tail when within <mtp_block_size_max tokens of context_tokens_max:
        // prevents speculative verify from wrapping global KV indices onto slots 0..6!
        if (use_mtp and !decode_stop_now and stats.generated_tokens < max_new_tokens and @as(u32, @intCast(self.current_position)) < self.context_tokens_max) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(&prev_fed_tok), @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            while (stats.generated_tokens < max_new_tokens and @as(u32, @intCast(self.current_position)) < self.context_tokens_max) {
                self.record_target_forward(1, true, true, false);
                cuda.squid_launch_inc_pos_and_copy_tok(self.d_pos, &self.d_tokens[0], &self.d_next_tokens[0], self.stream);
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_accepted.?), @ptrCast(self.d_next_tokens), @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

                if (self.prefix_cache.enabled and shield_tokens == 0 and self.prefix_cache.session_len < self.context_tokens_max) {
                    self.prefix_cache.session_tokens[self.prefix_cache.session_len] = prev_fed_tok;
                    self.prefix_cache.session_len += 1;
                }
                const tok: i32 = self.h_accepted.?[0];
                prev_fed_tok = tok;
                if (out_tokens.len > 0) out_tokens[stats.generated_tokens] = tok;
                stats.generated_tokens += 1;
                stats.last_token_id = tok;
                self.current_position += 1;
                if (on_token) |cb| {
                    if (!cb(callback_context.?, tok)) break;
                }
                if (tok == constants.eos_token_id) break;
            }
        }
        stats.decode_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - decode_start_ns)) / 1_000_000.0;

        if (self.prefix_cache.enabled and shield_tokens == 0) {
            self.prefix_cache.gpu_live_at_pos = self.current_position;
            if (self.prefix_cache.session_len >= 128) {
                self.save_local_state_checkpoint(3, self.prefix_cache.session_len);
            }
        }
        return stats;
    }

    fn prefill_range(self: *Engine, prompt: []const i32, start_i: u32, end_i: u32, use_mtp: bool, compute_last_lm: bool) u32 {
        tiger.assert(end_i >= start_i);
        var i: u32 = start_i;
        // Fast batched Tensor Core + multi-token MoE prefill (chunks of 32..256, multiples of 8)
        while (end_i - i >= 32 and (!compute_last_lm or end_i - i > 32)) {
            const avail: u32 = if (compute_last_lm) (end_i - i - 1) else (end_i - i);
            const chunk: u32 = @min(prefill_batch_max, avail & ~@as(u32, 7));
            if (chunk < 32) break;
            self.prefill_chunk_batched(prompt, i, chunk, use_mtp);
            i += chunk;
        }
        if (self.use_graphs and end_i - i >= constants.mtp_block_size_max) {
            const pgraph = self.prefill_b8_graph_exec[if (use_mtp) 1 else 0].?;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            while (i + constants.mtp_block_size_max <= end_i and (!compute_last_lm or i + constants.mtp_block_size_max < end_i)) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                    @ptrCast(self.d_tokens),
                    @ptrCast(prompt[i..].ptr),
                    constants.mtp_block_size_max * @sizeOf(i32),
                    self.stream,
                ));
                tiger.assert_cuda(cuda.squid_cuda_graph_launch(pgraph, self.stream));
                self.current_position += constants.mtp_block_size_max;
                i += constants.mtp_block_size_max;
            }
        }
        while (i < end_i) {
            const batch: u32 = @intCast(tiger.min_u64(constants.mtp_block_size_max, end_i - i));
            const is_last: bool = compute_last_lm and (i + batch == end_i);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(prompt[i..].ptr), batch * @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            self.record_target_forward(@intCast(batch), use_mtp, is_last, false);
            self.current_position += @intCast(batch);
            i += batch;
        }
        return i;
    }

    fn prefill_final_range(self: *Engine, prompt: []const i32, start_i: u32, prompt_len: u32, use_mtp: bool) u32 {
        tiger.assert(start_i < prompt_len);
        var i: u32 = start_i;
        // Fast batched Tensor Core + multi-token MoE prefill leaving 1..31 tokens for final lm_head chunk
        while (prompt_len - i > 32) {
            const avail: u32 = prompt_len - i - 1;
            const chunk: u32 = @min(prefill_batch_max, avail & ~@as(u32, 7));
            if (chunk < 32) break;
            self.prefill_chunk_batched(prompt, i, chunk, use_mtp);
            i += chunk;
        }
        if (self.use_graphs and prompt_len - i > constants.mtp_block_size_max) {
            const pgraph = self.prefill_b8_graph_exec[if (use_mtp) 1 else 0].?;
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            while (i + constants.mtp_block_size_max < prompt_len) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                    @ptrCast(self.d_tokens),
                    @ptrCast(prompt[i..].ptr),
                    constants.mtp_block_size_max * @sizeOf(i32),
                    self.stream,
                ));
                tiger.assert_cuda(cuda.squid_cuda_graph_launch(pgraph, self.stream));
                self.current_position += constants.mtp_block_size_max;
                i += constants.mtp_block_size_max;
            }
        }
        var last_batch: u32 = 1;
        while (i < prompt_len) {
            const batch: u32 = @intCast(tiger.min_u64(constants.mtp_block_size_max, prompt_len - i));
            const is_last: bool = (i + batch == prompt_len);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(prompt[i..].ptr), batch * @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            self.record_target_forward(@intCast(batch), use_mtp, is_last, false);
            self.current_position += @intCast(batch);
            i += batch;
            last_batch = batch;
        }
        return last_batch;
    }

    /// Execute one batched prefill chunk of `chunk_len` tokens (multiple of 8, 32..256).
    /// Uses cuBLAS FP16 Tensor Cores for dense/router GEMMs and multi-token half2 MoE kernels,
    /// while keeping sconv/attention/depth-norm/dspark_commit 100% exact via 8-token sub-slices.
    fn prefill_chunk_batched(self: *Engine, prompt: []const i32, start_i: u32, chunk_len: u32, commit_dspark: bool) void {
        tiger.assert(chunk_len >= 8 and chunk_len <= prefill_batch_max and (chunk_len % 8 == 0));
        const w = self.weights.?;
        const b_i32: i32 = @intCast(chunk_len);
        const b_usize: usize = chunk_len;
        const n_sub: usize = chunk_len / 8;

        var h_pos_vec: [prefill_sub_chunks_max]i32 = undefined;
        var s: usize = 0;
        while (s < n_sub) : (s += 1) {
            h_pos_vec[s] = self.current_position + @as(i32, @intCast(s * 8));
        }
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.p_pos_vec), @ptrCast(&h_pos_vec), n_sub * @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.p_tokens), @ptrCast(prompt[start_i..].ptr), b_usize * @sizeOf(i32), self.stream));

        cuda.squid_launch_embed_norm(w.embed.?, w.embed_norm.?, self.p_tokens, self.p_hidden, b_i32, self.stream);

        var block_count: i32 = 0;
        var has_prior: bool = false;
        var prev_mlp_pending: bool = false;

        const prof_on: bool = (tiger.getenv("SQUID_PROF_PREFILL") != null and start_i == 0);
        var t_pre: u64 = 0;
        var t_dense: u64 = 0;
        var t_attn: u64 = 0;
        var t_post: u64 = 0;
        var t_moe: u64 = 0;
        var t_commit: u64 = 0;
        var t_mark: u64 = if (prof_on) blk: {
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
            break :blk tiger.clock_monotonic_ns();
        } else 0;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const v = self.schedule[vi];
            const l: u32 = v.physical_layer;
            const lw: *const weights_mod.LayerWeights = &w.layers[l];

            var capture_prelude: bool = false;
            if (v.is_loop and l == constants.loop_start_layer) {
                if (v.pass_index == 0) {
                    capture_prelude = true;
                } else {
                    cuda.squid_launch_extract_block(self.p_blocks, 4, 2, self.p_saved, b_i32, self.stream);
                    cuda.squid_launch_rmsnorm(self.p_hidden, null, self.p_prior_last, b_i32, constants.hidden_dim, constants.hidden_dim, constants.rms_eps, self.stream);
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2d_async(@ptrCast(self.p_hidden), @ptrCast(self.p_prelude), b_usize * constants.hidden_dim * 4, self.stream));
                    block_count = 1;
                    has_prior = true;
                }
            }
            const use_prior: bool = v.is_loop and has_prior;
            const tap_slot: i32 = if (!commit_dspark) -1 else if (vi == constants.mtp_target_visits[0]) 0 else if (vi == constants.mtp_target_visits[1]) 1 else if (vi == constants.mtp_target_visits[2]) 2 else -1;

            const distinct_inputs: bool = v.is_global or use_prior;
            const is_block_start: bool = (l % constants.block_layer_count == 0);

            // 1. BATCHED Pre-attention sconv + depth-norm + optional cross-pass controller (B = chunk_len!)
            if (prev_mlp_pending) {
                const prev_lw: *const weights_mod.LayerWeights = &w.layers[self.schedule[vi - 1].physical_layer];
                const abl = ablation_slice(w, true, vi - 1);
                cuda.squid_launch_prefill_sconv_post_res(
                    self.p_moe_slot_out,
                    1,
                    prev_lw.mlp_sconv_w.?,
                    self.mlp_ring[vi - 1],
                    self.p_residual,
                    self.p_hidden,
                    if (capture_prelude) self.p_prelude else null,
                    null,
                    null,
                    0,
                    0,
                    b_i32,
                    &self.p_pos_vec[0],
                    abl.r,
                    abl.d,
                    abl.p,
                    self.stream,
                );
                if (tap_slot >= 0) {
                    cuda.squid_launch_prefill_copy_taps(self.p_hidden, self.p_taps, tap_slot, b_i32, self.stream);
                }
            }
            prev_mlp_pending = false;

            var n0: ?*const anyopaque = null;
            var n1: ?*const anyopaque = null;
            var n2: ?*const anyopaque = null;
            var o0: ?[*]f32 = null;
            var o1: ?[*]f32 = null;
            var o2: ?[*]f32 = null;
            var oh0: ?*anyopaque = null;
            var oh1: ?*anyopaque = null;
            var oh2: ?*anyopaque = null;
            var n_out: i32 = 1;

            if (!distinct_inputs) {
                n0 = lw.attn_norm;
                o0 = self.p_attn_in[0];
                oh0 = self.p_attn_in_h[0];
                n_out = 1;
            } else if (v.is_global and !use_prior) {
                n0 = lw.q_in_norm;
                n1 = lw.k_in_norm;
                n2 = lw.v_in_norm;
                o0 = self.p_attn_in[0];
                o1 = self.p_attn_in[1];
                o2 = self.p_attn_in[2];
                oh0 = self.p_attn_in_h[0];
                oh1 = self.p_attn_in_h[1];
                oh2 = self.p_attn_in_h[2];
                n_out = 3;
            } else if (v.is_global) {
                o0 = self.p_attn_in[0];
                o1 = self.p_attn_in[1];
                o2 = self.p_attn_in[2];
                n_out = 3;
            } else {
                o0 = self.p_mlp_in;
                n_out = 1;
            }

            cuda.squid_launch_fused_depth_norm(
                self.p_hidden,
                if (is_block_start) self.p_partial else null,
                self.p_blocks,
                block_count,
                if (l > 0) w.attn_queries[l - 1] else null,
                n0,
                n1,
                n2,
                o0,
                o1,
                o2,
                oh0,
                oh1,
                oh2,
                n_out,
                b_i32,
                self.stream,
            );

            if (use_prior) {
                const loc: usize = l - constants.loop_start_layer;
                const cp_q: [3][*]const f32 = .{
                    w.cross_pass_queries.? + (2 * loc) * constants.hidden_dim,
                    w.cross_pass_kv_queries.? + (2 * loc) * constants.hidden_dim,
                    w.cross_pass_kv_queries.? + (2 * loc + 1) * constants.hidden_dim,
                };
                const pnorms: [3]*const anyopaque = .{
                    (if (v.is_global) lw.q_in_norm else lw.attn_norm).?,
                    (if (v.is_global) lw.k_in_norm else lw.attn_norm).?,
                    (if (v.is_global) lw.v_in_norm else lw.attn_norm).?,
                };
                var cs: usize = 0;
                while (cs < 3) : (cs += 1) {
                    const x_in = if (v.is_global) self.p_attn_in[cs] else self.p_mlp_in;
                    cuda.squid_launch_prefill_controller_one(
                        x_in,
                        self.p_saved,
                        self.p_prior_last,
                        cp_q[cs],
                        &lw.ctrl[cs],
                        pnorms[cs],
                        self.p_attn_in[cs],
                        self.p_attn_in_h[cs],
                        self.p_ctrl_msg,
                        self.p_ctrl_cat_h,
                        self.p_ctrl_d,
                        self.p_ctrl_d_h,
                        self.p_ctrl_f,
                        self.p_ctrl_g,
                        self.p_w_f16,
                        b_i32,
                        self.stream,
                    );
                }
            }
            if (prof_on) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                const now = tiger.clock_monotonic_ns();
                t_pre += now - t_mark;
                t_mark = now;
            }

            // 2. BATCHED cuBLAS FP16 Tensor Core Attention Projections (B = chunk_len!)
            const in_qh = self.p_attn_in_h[0];
            const in_kh = if (distinct_inputs) self.p_attn_in_h[1] else self.p_attn_in_h[0];
            const in_vh = if (distinct_inputs) self.p_attn_in_h[2] else self.p_attn_in_h[0];
            cuda.squid_launch_prefill_gemm_h(&lw.w_qrgs, in_qh, self.p_qrgs, self.p_w_f16, b_i32, self.stream);
            cuda.squid_launch_prefill_gemm_h(&lw.w_k, in_kh, self.p_k_proj, self.p_w_f16, b_i32, self.stream);
            cuda.squid_launch_prefill_gemm_h(&lw.w_v, in_vh, self.p_v_proj, self.p_w_f16, b_i32, self.stream);
            if (prof_on) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                const now = tiger.clock_monotonic_ns();
                t_dense += now - t_mark;
                t_mark = now;
            }

            // 3. BATCHED Causal sconv + KV store + Blackwell Tensor Core FlashAttention!
            // Global layers: 1-shot batched B<=512 (Q_TOK=32 -> 96 queries/CTA, 128 CTAs = 1 wave!).
            // Local layers: B_sub<=256 (Q_TOK=16 -> 48 queries/CTA, 128 CTAs = 1 wave, 100% ring-safe with kv_cap=1280!).
            if (v.is_global) {
                cuda.squid_launch_prefill_global_sconv_prep_store(
                    self.p_qrgs,
                    self.p_k_proj,
                    self.p_v_proj,
                    in_kh,
                    in_vh,
                    lw.k_sconv_w.?,
                    lw.v_sconv_w.?,
                    lw.k_sconv_proj.?,
                    lw.v_sconv_proj.?,
                    self.k_ring[vi],
                    self.v_ring[vi],
                    lw.q_clip_scale.?,
                    lw.rel_clip_scale.?,
                    self.k_cache[vi].?,
                    self.v_cache[vi].?,
                    self.k_scale[vi],
                    self.v_scale[vi],
                    @intFromEnum(self.kv_format),
                    self.kv_cap[vi],
                    self.p_ctrl_d,
                    self.p_w_f16,
                    b_i32,
                    &self.p_pos_vec[0],
                    self.stream,
                );
                const share_bulk: bool = self.share_loop_kv and is_shared_loop_global_visit(vi);
                const bulk_vi: u32 = shared_bulk_visit_idx(vi);
                cuda.squid_launch_prefill_flash_attn_tc(
                    self.p_qrgs,
                    lw.rel_logits_proj.?,
                    self.k_cache[vi].?,
                    self.v_cache[vi].?,
                    self.k_scale[vi],
                    self.v_scale[vi],
                    if (share_bulk) self.k_cache[bulk_vi] else null,
                    if (share_bulk) self.v_cache[bulk_vi] else null,
                    if (share_bulk) self.k_scale[bulk_vi] else null,
                    if (share_bulk) self.v_scale[bulk_vi] else null,
                    if (share_bulk) self.kv_cap[bulk_vi] else 0,
                    @intFromEnum(self.kv_format),
                    self.p_attn_gated_h,
                    self.kv_cap[vi],
                    0,
                    b_i32,
                    &self.p_pos_vec[0],
                    self.stream,
                );
            } else {
                var off_u32: u32 = 0;
                while (off_u32 < chunk_len) {
                    const sub_b: u32 = @min(prefill_local_attn_sub, chunk_len - off_u32);
                    const off: usize = off_u32;
                    const d_pos_sub: *const i32 = &self.p_pos_vec[off / 8];
                    const qrgs_sub = self.p_qrgs + off * constants.qrgs_out_dim;
                    const k_sub = self.p_k_proj + off * constants.kv_dim;
                    const v_sub = self.p_v_proj + off * constants.kv_dim;

                    cuda.squid_launch_prefill_local_sconv_prep_store(
                        qrgs_sub,
                        k_sub,
                        v_sub,
                        lw.k_sconv_w.?,
                        lw.v_sconv_w.?,
                        self.k_ring[vi],
                        self.v_ring[vi],
                        lw.q_clip_scale.?,
                        lw.rel_clip_scale.?,
                        self.k_cache[vi].?,
                        self.v_cache[vi].?,
                        self.kv_cap[vi],
                        @intCast(sub_b),
                        d_pos_sub,
                        self.stream,
                    );

                    cuda.squid_launch_prefill_flash_attn_tc(
                        qrgs_sub,
                        lw.rel_logits_proj.?,
                        self.k_cache[vi].?,
                        self.v_cache[vi].?,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        0,
                        @intFromEnum(constants.KVFormat.bf16),
                        offset_f16_ptr(self.p_attn_gated_h, off * constants.query_dim),
                        self.kv_cap[vi],
                        constants.swa_window_tokens,
                        @intCast(sub_b),
                        d_pos_sub,
                        self.stream,
                    );
                    off_u32 += sub_b;
                }
            }
            if (prof_on) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                const now = tiger.clock_monotonic_ns();
                t_attn += now - t_mark;
                t_mark = now;
            }

            // 4. BATCHED cuBLAS FP16 Tensor Core O Projection (B = chunk_len!)
            cuda.squid_launch_prefill_gemm_h(&lw.w_o, self.p_attn_gated_h, self.p_o_proj, self.p_w_f16, b_i32, self.stream);
            if (prof_on) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                const now = tiger.clock_monotonic_ns();
                t_dense += now - t_mark;
                t_mark = now;
            }

            // 5. BATCHED Attn post sconv + residual + MLP depth-norm (+ optional MLP controller) (B = chunk_len!)
            const is_boundary: bool = (l % constants.block_layer_count == 0);
            const attn_abl = ablation_slice(w, false, vi);
            cuda.squid_launch_prefill_sconv_post_res(
                self.p_o_proj,
                0,
                lw.attn_sconv_w.?,
                self.attn_ring[vi],
                self.p_hidden,
                self.p_residual,
                null,
                self.p_blocks,
                self.p_partial,
                block_count,
                if (is_boundary) 1 else 0,
                b_i32,
                &self.p_pos_vec[0],
                attn_abl.r,
                attn_abl.d,
                attn_abl.p,
                self.stream,
            );
            if (is_boundary) block_count += 1;

            cuda.squid_launch_fused_depth_norm(
                self.p_residual,
                null,
                self.p_blocks,
                block_count,
                w.mlp_queries[l],
                if (use_prior) null else lw.mlp_norm,
                null,
                null,
                if (use_prior) self.p_mlp_in else self.p_mlp_normed,
                null,
                null,
                if (use_prior) null else self.p_mlp_normed_h,
                null,
                null,
                1,
                b_i32,
                self.stream,
            );

            if (use_prior) {
                const loc: usize = l - constants.loop_start_layer;
                cuda.squid_launch_prefill_controller_one(
                    self.p_mlp_in,
                    self.p_saved,
                    self.p_prior_last,
                    w.cross_pass_queries.? + (2 * loc + 1) * constants.hidden_dim,
                    &lw.ctrl[3],
                    lw.mlp_norm.?,
                    self.p_mlp_normed,
                    self.p_mlp_normed_h,
                    self.p_ctrl_msg,
                    self.p_ctrl_cat_h,
                    self.p_ctrl_d,
                    self.p_ctrl_d_h,
                    self.p_ctrl_f,
                    self.p_ctrl_g,
                    self.p_w_f16,
                    b_i32,
                    self.stream,
                );
            }
            if (prof_on) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                const now = tiger.clock_monotonic_ns();
                t_post += now - t_mark;
                t_mark = now;
            }

            // 6. BATCHED MoE Router (cuBLAS TC) + Counting Sort + Multi-Token MoE w13 & w2 (B = chunk_len!)
            const router_lw: cuda.CLinearWeight = if (w.mode == .nvfp4)
                lw.router_nvfp4
            else
                .{
                    .kind = @intFromEnum(constants.WeightKind.bf16),
                    .rows = constants.expert_count,
                    .cols = constants.hidden_dim,
                    .data = lw.router_w,
                };
            cuda.squid_launch_prefill_gemm_h(&router_lw, self.p_mlp_normed_h, self.p_router_logits, self.p_w_f16, b_i32, self.stream);
            cuda.squid_launch_prefill_route_and_sort(
                self.p_router_logits,
                lw.router_bias.?,
                self.p_route_ids,
                self.p_route_w,
                self.p_exp_offsets,
                self.p_sorted_tok,
                self.p_sorted_slot,
                self.p_sorted_w,
                b_i32,
                self.stream,
            );
            cuda.squid_launch_prefill_moe_gate_up_h(
                &lw.w13,
                self.p_mlp_normed_h,
                self.p_exp_offsets,
                self.p_sorted_tok,
                self.p_sorted_slot,
                self.p_moe_act_h,
                b_i32,
                self.stream,
            );
            cuda.squid_launch_prefill_moe_down_h(
                &lw.w2,
                self.p_moe_act_h,
                self.p_exp_offsets,
                self.p_sorted_tok,
                self.p_sorted_slot,
                self.p_sorted_w,
                self.p_moe_slot_out,
                b_i32,
                self.stream,
            );
            if (prof_on) {
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
                const now = tiger.clock_monotonic_ns();
                t_moe += now - t_mark;
                t_mark = now;
            }

            if (vi == 17 or vi == 29) {
                const mlp_abl = ablation_slice(w, true, vi);
                cuda.squid_launch_prefill_sconv_post_res(
                    self.p_moe_slot_out,
                    1,
                    lw.mlp_sconv_w.?,
                    self.mlp_ring[vi],
                    self.p_residual,
                    self.p_hidden,
                    null,
                    null,
                    null,
                    0,
                    0,
                    b_i32,
                    &self.p_pos_vec[0],
                    mlp_abl.r,
                    mlp_abl.d,
                    mlp_abl.p,
                    self.stream,
                );
                prev_mlp_pending = false;
            } else {
                prev_mlp_pending = true;
            }
        }

        // Visit 47 final MLP post + DSpark commit
        const last_lw = &w.layers[self.schedule[constants.visit_count - 1].physical_layer];
        const last_mlp_abl = ablation_slice(w, true, constants.visit_count - 1);
        cuda.squid_launch_prefill_sconv_post_res(
            self.p_moe_slot_out,
            1,
            last_lw.mlp_sconv_w.?,
            self.mlp_ring[constants.visit_count - 1],
            self.p_residual,
            self.p_hidden,
            null,
            null,
            null,
            0,
            0,
            b_i32,
            &self.p_pos_vec[0],
            last_mlp_abl.r,
            last_mlp_abl.d,
            last_mlp_abl.p,
            self.stream,
        );
        cuda.squid_launch_fused_depth_norm(
            self.p_hidden,
            null,
            self.p_blocks,
            block_count,
            w.output_query,
            w.final_norm,
            null,
            null,
            self.p_mlp_normed,
            null,
            null,
            null,
            null,
            null,
            1,
            b_i32,
            self.stream,
        );
        if (commit_dspark) {
            cuda.squid_launch_prefill_copy_taps(self.p_mlp_normed, self.p_taps, 3, b_i32, self.stream);
            s = 0;
            while (s < n_sub) : (s += 1) {
                const d_pos_sub: *const i32 = &self.p_pos_vec[s];
                cuda.squid_launch_dspark_commit(
                    &w.dspark,
                    self.p_taps + (s * 4) * 8 * constants.hidden_dim,
                    &self.ds_k_cache,
                    &self.ds_v_cache,
                    &self.ds_k_scale,
                    &self.ds_v_scale,
                    &self.ds_kv_cap,
                    @intFromEnum(self.kv_format),
                    self.ds_f_ring.?,
                    self.ds_mix,
                    self.ds_kv,
                    8,
                    d_pos_sub,
                    self.stream,
                );
            }
        }
        if (prof_on) {
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
            t_commit = tiger.clock_monotonic_ns() - t_mark;
            const tot_ms = @as(f64, @floatFromInt(t_pre + t_dense + t_attn + t_post + t_moe + t_commit)) / 1e6;
            tiger.print_stdout(
                "[prof-prefill B={d}] total={d:.2}ms ({d:.0} tok/s) | pre_sub={d:.2}ms dense_tc={d:.2}ms attn_sub={d:.2}ms post_sub={d:.2}ms moe={d:.2}ms commit={d:.2}ms\n",
                .{
                    chunk_len,
                    tot_ms,
                    @as(f64, @floatFromInt(chunk_len)) * 1000.0 / tot_ms,
                    @as(f64, @floatFromInt(t_pre)) / 1e6,
                    @as(f64, @floatFromInt(t_dense)) / 1e6,
                    @as(f64, @floatFromInt(t_attn)) / 1e6,
                    @as(f64, @floatFromInt(t_post)) / 1e6,
                    @as(f64, @floatFromInt(t_moe)) / 1e6,
                    @as(f64, @floatFromInt(t_commit)) / 1e6,
                },
            );
        }
        self.current_position += b_i32;
    }

    /// Long-context latency microbenchmark at simulated KV position `sim_pos`.
    pub fn bench_step_at_pos(self: *Engine, sim_pos_in: i32, num_drafts: u32, iterations: u32) f64 {
        tiger.assert(iterations > 0 and iterations <= 256);
        const w = self.weights.?;
        const use_mtp: bool = (num_drafts > 0 and w.dspark.loaded != 0);
        const max_allowed: i32 = @as(i32, @intCast(self.context_tokens_max)) - @as(i32, @intCast(iterations * constants.mtp_block_size_max)) - 32;
        var sim_pos: i32 = sim_pos_in;
        if (sim_pos > max_allowed) sim_pos = tiger.max_i32(0, max_allowed);

        self.reset();

        // Zero-init global KV caches so synthetic reads are strictly finite (zero NaN)
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            if (!self.schedule[vi].is_global) continue;
            const cap: u64 = @intCast(self.kv_cap[vi]);
            const data_bytes: usize = @intCast(switch (self.kv_format) {
                .bf16 => cap * constants.kv_dim * 2,
                .fp8 => cap * constants.kv_dim,
                .q4 => cap * (constants.kv_dim / 2),
            });
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.k_cache[vi].?, 0, data_bytes, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.v_cache[vi].?, 0, data_bytes, self.stream));
            if (self.k_scale[vi]) |ks| {
                const scale_bytes: usize = @intCast(if (self.kv_format == .fp8)
                    cap * constants.kv_head_count * 2
                else
                    cap * constants.kv_q4_scales_per_token * 2);
                tiger.assert_cuda(cuda.squid_cuda_memset_async(ks, 0, scale_bytes, self.stream));
                tiger.assert_cuda(cuda.squid_cuda_memset_async(self.v_scale[vi].?, 0, scale_bytes, self.stream));
            }
        }
        if (use_mtp) {
            const cap: u64 = @intCast(self.ds_kv_cap[1]);
            const data_bytes: usize = @intCast(switch (self.kv_format) {
                .bf16 => cap * constants.kv_dim * 2,
                .fp8 => cap * constants.kv_dim,
                .q4 => cap * (constants.kv_dim / 2),
            });
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.ds_k_cache[1].?, 0, data_bytes, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.ds_v_cache[1].?, 0, data_bytes, self.stream));
            if (self.ds_k_scale[1]) |ks| {
                const scale_bytes: usize = @intCast(if (self.kv_format == .fp8)
                    cap * constants.kv_head_count * 2
                else
                    cap * constants.kv_q4_scales_per_token * 2);
                tiger.assert_cuda(cuda.squid_cuda_memset_async(ks, 0, scale_bytes, self.stream));
                tiger.assert_cuda(cuda.squid_cuda_memset_async(self.ds_v_scale[1].?, 0, scale_bytes, self.stream));
            }
        }

        const seed_tok: i32 = 150_000;
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(&seed_tok), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_anchor), @ptrCast(&seed_tok), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&sim_pos), @sizeOf(i32), self.stream));

        if (!use_mtp) self.ensure_b1_graph() else self.ensure_spec_graph(num_drafts);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        const graph_exec = if (use_mtp) self.spec_graph_exec[num_drafts].? else self.b1_graph_exec.?;
        var warmup: u32 = 0;
        while (warmup < 4) : (warmup += 1) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&sim_pos), @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_graph_launch(graph_exec, self.stream));
        }
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&sim_pos), @sizeOf(i32), self.stream));
        var ev0: ?*anyopaque = null;
        var ev1: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_event_create(&ev0, 0));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&ev1, 0));
        defer tiger.assert_cuda(cuda.squid_cuda_event_destroy(ev0));
        defer tiger.assert_cuda(cuda.squid_cuda_event_destroy(ev1));

        tiger.assert_cuda(cuda.squid_cuda_event_record(ev0, self.stream));
        var it: u32 = 0;
        while (it < iterations) : (it += 1) {
            tiger.assert_cuda(cuda.squid_cuda_graph_launch(graph_exec, self.stream));
        }
        tiger.assert_cuda(cuda.squid_cuda_event_record(ev1, self.stream));
        tiger.assert_cuda(cuda.squid_cuda_event_sync(ev1));

        var elapsed_ms: f32 = 0.0;
        tiger.assert_cuda(cuda.squid_cuda_event_elapsed_ms(&elapsed_ms, ev0, ev1));
        return @as(f64, @floatCast(elapsed_ms)) / @as(f64, @floatFromInt(iterations));
    }

    fn alloc_kv_pair(
        self: *Engine,
        cap_tokens: u32,
        kv_format: constants.KVFormat,
        k_out: *?*anyopaque,
        v_out: *?*anyopaque,
        ks_out: *?*anyopaque,
        vs_out: *?*anyopaque,
    ) void {
        const cap: u64 = cap_tokens;
        switch (kv_format) {
            .bf16 => {
                const bytes = cap * constants.kv_dim * 2;
                k_out.* = self.arena.alloc_raw(bytes);
                v_out.* = self.arena.alloc_raw(bytes);
                ks_out.* = null;
                vs_out.* = null;
            },
            .fp8 => {
                const data_bytes = cap * constants.kv_dim;
                const scale_bytes = cap * constants.kv_head_count * 2;
                k_out.* = self.arena.alloc_raw(data_bytes);
                v_out.* = self.arena.alloc_raw(data_bytes);
                ks_out.* = self.arena.alloc_raw(scale_bytes);
                vs_out.* = self.arena.alloc_raw(scale_bytes);
            },
            .q4 => {
                const data_bytes = cap * (constants.kv_dim / 2);
                const scale_bytes = cap * constants.kv_q4_scales_per_token * 2;
                k_out.* = self.arena.alloc_raw(data_bytes);
                v_out.* = self.arena.alloc_raw(data_bytes);
                ks_out.* = self.arena.alloc_raw(scale_bytes);
                vs_out.* = self.arena.alloc_raw(scale_bytes);
            },
        }
    }

    pub const LogitsChunkCallbackFn = *const fn (context: *anyopaque, pos_offset: usize, chunk_logits: []const f32, count: usize) bool;

    /// Stream exact post-softcap FP32 logits in 8-token chunks (6.2 MiB host RAM regardless of context length!).
    pub fn eval_logits_streaming(
        self: *Engine,
        tokens: []const i32,
        chunk_buf: []f32,
        on_chunk: LogitsChunkCallbackFn,
        callback_context: *anyopaque,
    ) bool {
        const vocab: usize = constants.vocab_unpadded_count;
        tiger.assert(tokens.len > 0 and tokens.len < self.context_tokens_max);
        tiger.assert(chunk_buf.len >= constants.mtp_block_size_max * vocab);
        self.reset();

        const dump_cfg: cuda.CSampleConfig = .{ .inv_temp = -1.0, .top_p = 1.0, .top_k = 0, .pad = 0, .seed = 0 };
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_sample_cfg), @ptrCast(&dump_cfg), @sizeOf(cuda.CSampleConfig), self.stream));
        defer {
            const zero_cfg: cuda.CSampleConfig = .{ .inv_temp = 0.0, .top_p = 1.0, .top_k = 0, .pad = 0, .seed = 0 };
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_sample_cfg), @ptrCast(&zero_cfg), @sizeOf(cuda.CSampleConfig), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
        }

        var i: usize = 0;
        while (i < tokens.len) {
            const batch: usize = @min(@as(usize, constants.mtp_block_size_max), tokens.len - i);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(tokens[i..].ptr), batch * @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            self.record_target_forward(@intCast(batch), false, true, false);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(
                @ptrCast(chunk_buf.ptr),
                @ptrCast(self.lm_logits),
                batch * vocab * @sizeOf(f32),
                self.stream,
            ));
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
            self.current_position += @intCast(batch);
            if (!on_chunk(callback_context, i, chunk_buf[0 .. batch * vocab], batch)) return false;
            i += batch;
        }
        return true;
    }

    /// Evaluate exact post-softcap FP32 logits [tokens.len, VOCAB_UNPADDED] for KLD / PPL quality eval.
    pub fn eval_logits_chunked(self: *Engine, tokens: []const i32, out_logits_host: []f32) void {
        const vocab: usize = constants.vocab_unpadded_count;
        tiger.assert(tokens.len > 0 and tokens.len < self.context_tokens_max);
        tiger.assert(out_logits_host.len >= tokens.len * vocab);
        self.reset();

        const dump_cfg: cuda.CSampleConfig = .{ .inv_temp = -1.0, .top_p = 1.0, .top_k = 0, .pad = 0, .seed = 0 };
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_sample_cfg), @ptrCast(&dump_cfg), @sizeOf(cuda.CSampleConfig), self.stream));
        defer {
            const zero_cfg: cuda.CSampleConfig = .{ .inv_temp = 0.0, .top_p = 1.0, .top_k = 0, .pad = 0, .seed = 0 };
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_sample_cfg), @ptrCast(&zero_cfg), @sizeOf(cuda.CSampleConfig), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
        }

        var i: usize = 0;
        while (i < tokens.len) {
            const batch: usize = @min(@as(usize, constants.mtp_block_size_max), tokens.len - i);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(tokens[i..].ptr), batch * @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            self.record_target_forward(@intCast(batch), false, true, false);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(
                @ptrCast(out_logits_host[i * vocab ..].ptr),
                @ptrCast(self.lm_logits),
                batch * vocab * @sizeOf(f32),
                self.stream,
            ));
            tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
            self.current_position += @intCast(batch);
            i += batch;
        }
    }

    fn record_target_forward(self: *Engine, batch_size: i32, commit_dspark: bool, compute_lm_head: bool, is_verify: bool) void {
        tiger.assert(batch_size >= 1 and batch_size <= constants.mtp_block_size_max);
        const w = self.weights.?;
        const b_usize: usize = @intCast(batch_size);
        // TC moe_down_tc_kernel sums all 8 expert slots in registers and writes [B, HIDDEN] directly into self.moe_down!
        const moe_slotted: u8 = 0;

        cuda.squid_launch_embed_norm(w.embed.?, w.embed_norm.?, self.d_tokens, self.hidden, batch_size, self.stream);

        var block_count: i32 = 0;
        var has_prior: bool = false;
        var prev_mlp_pending: bool = false;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const v = self.schedule[vi];
            const l: u32 = v.physical_layer;
            const lw: *const weights_mod.LayerWeights = &w.layers[l];

            var pre_tap_out: ?[*]f32 = null;
            if (v.is_loop and l == constants.loop_start_layer) {
                if (v.pass_index == 0) {
                    pre_tap_out = self.prelude;
                } else {
                    cuda.squid_launch_extract_block(self.blocks, 4, 2, self.saved, batch_size, self.stream);
                    cuda.squid_launch_rmsnorm(self.hidden, null, self.prior_last, batch_size, constants.hidden_dim, constants.hidden_dim, constants.rms_eps, self.stream);
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2d_async(@ptrCast(self.hidden), @ptrCast(self.prelude), b_usize * constants.hidden_dim * 4, self.stream));
                    block_count = 1;
                    has_prior = true;
                }
            }
            const use_prior: bool = v.is_loop and has_prior;

            if (commit_dspark) {
                if (vi == constants.mtp_target_visits[0]) {
                    pre_tap_out = self.taps;
                } else if (vi == constants.mtp_target_visits[1]) {
                    pre_tap_out = self.taps + b_usize * constants.hidden_dim;
                } else if (vi == constants.mtp_target_visits[2]) {
                    pre_tap_out = self.taps + 2 * b_usize * constants.hidden_dim;
                }
            }

            const distinct_inputs: bool = v.is_global or use_prior;
            const partial_dst: ?[*]f32 = if (l % constants.block_layer_count == 0) self.partial else null;
            const use_half_dense: bool = (lw.w_qrgs.kind != @intFromEnum(constants.WeightKind.bf16));

            var n0: ?*const anyopaque = null;
            var n1: ?*const anyopaque = null;
            var n2: ?*const anyopaque = null;
            var o0: ?[*]f32 = null;
            var o1: ?[*]f32 = null;
            var o2: ?[*]f32 = null;
            var oh0: ?*anyopaque = null;
            var oh1: ?*anyopaque = null;
            var oh2: ?*anyopaque = null;
            var n_out: i32 = 1;

            if (!distinct_inputs) {
                n0 = lw.attn_norm;
                o0 = self.attn_in[0];
                oh0 = if (use_half_dense) self.attn_in_h[0] else null;
                n_out = 1;
            } else if (v.is_global and !use_prior) {
                n0 = lw.q_in_norm;
                n1 = lw.k_in_norm;
                n2 = lw.v_in_norm;
                o0 = self.attn_in[0];
                o1 = self.attn_in[1];
                o2 = self.attn_in[2];
                oh0 = if (use_half_dense) self.attn_in_h[0] else null;
                oh1 = if (use_half_dense) self.attn_in_h[1] else null;
                oh2 = if (use_half_dense) self.attn_in_h[2] else null;
                n_out = 3;
            } else if (v.is_global) {
                o0 = self.attn_in[0];
                o1 = self.attn_in[1];
                o2 = self.attn_in[2];
                n_out = 3;
            } else {
                o0 = self.mlp_in;
                n_out = 1;
            }

            if (prev_mlp_pending) {
                const prev_lw: *const weights_mod.LayerWeights = &w.layers[self.schedule[vi - 1].physical_layer];
                const prev_abl = ablation_slice(w, true, vi - 1);
                cuda.squid_launch_mega_mlp_post_next_depth_norm(
                    self.moe_down,
                    moe_slotted,
                    prev_lw.mlp_sconv_w.?,
                    self.mlp_ring[vi - 1],
                    self.residual,
                    self.hidden,
                    pre_tap_out,
                    null,
                    partial_dst,
                    self.blocks,
                    block_count,
                    w.attn_queries[l - 1].?,
                    n0,
                    n1,
                    n2,
                    o0,
                    o1,
                    o2,
                    oh0,
                    oh1,
                    oh2,
                    n_out,
                    batch_size,
                    self.d_pos,
                    prev_abl.r,
                    prev_abl.d,
                    prev_abl.p,
                    self.stream,
                );
                prev_mlp_pending = false;
            } else {
                cuda.squid_launch_fused_depth_norm(
                    self.hidden,
                    partial_dst,
                    self.blocks,
                    block_count,
                    if (l > 0) w.attn_queries[l - 1] else null,
                    n0,
                    n1,
                    n2,
                    o0,
                    o1,
                    o2,
                    oh0,
                    oh1,
                    oh2,
                    n_out,
                    batch_size,
                    self.stream,
                );
            }

            if (use_prior) {
                const loc: usize = l - constants.loop_start_layer;
                const xs: [3][*]f32 = .{
                    if (v.is_global) self.attn_in[0] else self.mlp_in,
                    if (v.is_global) self.attn_in[1] else self.mlp_in,
                    if (v.is_global) self.attn_in[2] else self.mlp_in,
                };
                const outs: [3][*]f32 = .{ self.attn_in[0], self.attn_in[1], self.attn_in[2] };
                const outs_h: [3]?*anyopaque = .{ self.attn_in_h[0], self.attn_in_h[1], self.attn_in_h[2] };
                const cp_q: [3]?[*]const f32 = .{
                    w.cross_pass_queries.? + (2 * loc) * constants.hidden_dim,
                    w.cross_pass_kv_queries.? + (2 * loc) * constants.hidden_dim,
                    w.cross_pass_kv_queries.? + (2 * loc + 1) * constants.hidden_dim,
                };
                const cws: [3]?*const cuda.CControllerWeights = .{ &lw.ctrl[0], &lw.ctrl[1], &lw.ctrl[2] };
                const pnorms: [3]?*const anyopaque = .{
                    if (v.is_global) lw.q_in_norm else lw.attn_norm,
                    if (v.is_global) lw.k_in_norm else lw.attn_norm,
                    if (v.is_global) lw.v_in_norm else lw.attn_norm,
                };
                if (w.mode == .nvfp4) {
                    cuda.squid_launch_controller_mega_nvfp4(
                        &xs,
                        self.saved,
                        self.prior_last,
                        &cp_q,
                        &cws,
                        &pnorms,
                        &outs,
                        if (use_half_dense) &outs_h else null,
                        3,
                        batch_size,
                        self.stream,
                    );
                } else {
                    cuda.squid_launch_controller_merge_batched(
                        &xs,
                        self.saved,
                        self.prior_last,
                        &cp_q,
                        &cws,
                        &pnorms,
                        &outs,
                        if (use_half_dense) &outs_h else null,
                        3,
                        self.ctrl_msg,
                        self.ctrl_cat,
                        self.ctrl_d,
                        batch_size,
                        self.stream,
                    );
                }
            }

            const in_q = self.attn_in[0];
            const in_k = if (distinct_inputs) self.attn_in[1] else self.attn_in[0];
            const in_v = if (distinct_inputs) self.attn_in[2] else self.attn_in[0];
            const in_qh = self.attn_in_h[0];
            const in_kh = if (distinct_inputs) self.attn_in_h[1] else self.attn_in_h[0];
            const in_vh = if (distinct_inputs) self.attn_in_h[2] else self.attn_in_h[0];

            if (lw.w_qrgs.kind != @intFromEnum(constants.WeightKind.bf16)) {
                cuda.squid_launch_qkvrg_fused(&lw.w_qrgs, &lw.w_k, &lw.w_v, in_qh, in_kh, in_vh, self.qrgs, self.k_proj, self.v_proj, batch_size, self.stream);
            } else {
                cuda.squid_launch_linear(&lw.w_qrgs, in_q, self.qrgs, batch_size, 0, self.stream);
                cuda.squid_launch_linear(&lw.w_k, in_k, self.k_proj, batch_size, 0, self.stream);
                cuda.squid_launch_linear(&lw.w_v, in_v, self.v_proj, batch_size, 0, self.stream);
            }

            if (v.is_global) {
                cuda.squid_launch_mega_global_sconv_prep_store(
                    self.qrgs,
                    self.k_proj,
                    self.v_proj,
                    in_k,
                    in_v,
                    lw.k_sconv_w.?,
                    lw.v_sconv_w.?,
                    lw.k_sconv_proj.?,
                    lw.v_sconv_proj.?,
                    self.k_ring[vi],
                    self.v_ring[vi],
                    lw.q_clip_scale.?,
                    lw.rel_clip_scale.?,
                    self.k_cache[vi].?,
                    self.v_cache[vi].?,
                    self.k_scale[vi],
                    self.v_scale[vi],
                    @intFromEnum(self.kv_format),
                    self.kv_cap[vi],
                    batch_size,
                    self.d_pos,
                    self.stream,
                );
            } else {
                cuda.squid_launch_mega_local_sconv_prep_store(
                    self.qrgs,
                    self.k_proj,
                    self.v_proj,
                    lw.k_sconv_w.?,
                    lw.v_sconv_w.?,
                    self.k_ring[vi],
                    self.v_ring[vi],
                    lw.q_clip_scale.?,
                    lw.rel_clip_scale.?,
                    self.k_cache[vi].?,
                    self.v_cache[vi].?,
                    self.kv_cap[vi],
                    batch_size,
                    self.d_pos,
                    self.stream,
                );
            }

            const share_bulk: bool = v.is_global and self.share_loop_kv and is_shared_loop_global_visit(vi);
            const bulk_vi: u32 = shared_bulk_visit_idx(vi);
            cuda.squid_launch_relative_attention_gated(
                self.qrgs,
                lw.rel_logits_proj.?,
                self.k_cache[vi].?,
                self.v_cache[vi].?,
                self.k_scale[vi],
                self.v_scale[vi],
                if (share_bulk) self.k_cache[bulk_vi] else null,
                if (share_bulk) self.v_cache[bulk_vi] else null,
                if (share_bulk) self.k_scale[bulk_vi] else null,
                if (share_bulk) self.v_scale[bulk_vi] else null,
                if (share_bulk) self.kv_cap[bulk_vi] else 0,
                if (v.is_global) @intFromEnum(self.kv_format) else @intFromEnum(constants.KVFormat.bf16),
                self.split_buf,
                if (use_half_dense) null else self.attn_gated,
                if (use_half_dense) self.attn_gated_h else null,
                self.kv_cap[vi],
                if (v.is_global) 0 else constants.swa_window_tokens,
                batch_size,
                self.d_pos,
                self.stream,
            );

            if (use_half_dense) {
                cuda.squid_launch_linear_h(&lw.w_o, self.attn_gated_h, self.o_proj, batch_size, 0, self.stream);
            } else {
                cuda.squid_launch_linear(&lw.w_o, self.attn_gated, self.o_proj, batch_size, 0, self.stream);
            }

            const is_boundary: bool = (l % constants.block_layer_count == 0);
            const attn_abl = ablation_slice(w, false, vi);
            cuda.squid_launch_mega_attn_post_mlp_in(
                self.o_proj,
                lw.attn_sconv_w.?,
                self.attn_ring[vi],
                self.hidden,
                self.partial,
                self.blocks,
                block_count,
                if (is_boundary) 1 else 0,
                self.residual,
                w.mlp_queries[l].?,
                if (use_prior) null else lw.mlp_norm,
                if (use_prior) self.mlp_in else self.mlp_normed,
                if (use_prior) null else self.mlp_normed_h,
                batch_size,
                self.d_pos,
                attn_abl.r,
                attn_abl.d,
                attn_abl.p,
                self.stream,
            );
            if (is_boundary) block_count += 1;

            if (use_prior) {
                const loc: usize = l - constants.loop_start_layer;
                const xs1: [1][*]f32 = .{self.mlp_in};
                const outs1: [1][*]f32 = .{self.mlp_normed};
                const outs_h1: [1]?*anyopaque = .{self.mlp_normed_h};
                const cp_q1: [3]?[*]const f32 = .{ w.cross_pass_queries.? + (2 * loc + 1) * constants.hidden_dim, null, null };
                const cws1: [3]?*const cuda.CControllerWeights = .{ &lw.ctrl[3], null, null };
                const pnorms1: [3]?*const anyopaque = .{ lw.mlp_norm, null, null };
                if (w.mode == .nvfp4) {
                    cuda.squid_launch_controller_mega_nvfp4(&xs1, self.saved, self.prior_last, &cp_q1, &cws1, &pnorms1, &outs1, &outs_h1, 1, batch_size, self.stream);
                } else {
                    cuda.squid_launch_controller_merge_batched(&xs1, self.saved, self.prior_last, &cp_q1, &cws1, &pnorms1, &outs1, &outs_h1, 1, self.ctrl_msg, self.ctrl_cat, self.ctrl_d, batch_size, self.stream);
                }
            }

            if (w.mode == .nvfp4) {
                cuda.squid_launch_route_topk_nvfp4(
                    self.mlp_normed_h,
                    &lw.router_nvfp4,
                    lw.router_bias.?,
                    self.router_logits,
                    self.route_ids,
                    self.route_w,
                    null,
                    batch_size,
                    self.stream,
                );
            } else {
                cuda.squid_launch_route_topk(
                    self.mlp_normed,
                    lw.router_w.?,
                    lw.router_bias.?,
                    self.router_logits,
                    self.route_ids,
                    self.route_w,
                    null,
                    batch_size,
                    self.stream,
                );
            }
            cuda.squid_launch_moe_gate_up_h(&lw.w13, self.mlp_normed_h, self.route_ids, self.exp_groups, self.moe_act_h, batch_size, self.stream);
            cuda.squid_launch_moe_down_h(&lw.w2, self.moe_act_h, self.route_ids, self.route_w, self.exp_groups, self.moe_down, batch_size, self.stream);

            if (vi == 17 or vi == 29) {
                const mlp_abl = ablation_slice(w, true, vi);
                cuda.squid_launch_mega_mlp_post_res(
                    self.moe_down,
                    moe_slotted,
                    lw.mlp_sconv_w.?,
                    self.mlp_ring[vi],
                    self.residual,
                    self.hidden,
                    batch_size,
                    self.d_pos,
                    mlp_abl.r,
                    mlp_abl.d,
                    mlp_abl.p,
                    self.stream,
                );
                prev_mlp_pending = false;
            } else {
                prev_mlp_pending = true;
            }
        }

        // Visit 47 final MLP post + output depth attention + final norm (+ tap 3)
        const last_mlp_abl = ablation_slice(w, true, constants.visit_count - 1);
        cuda.squid_launch_mega_mlp_post_next_depth_norm(
            self.moe_down,
            moe_slotted,
            w.layers[self.schedule[constants.visit_count - 1].physical_layer].mlp_sconv_w.?,
            self.mlp_ring[constants.visit_count - 1],
            self.residual,
            self.hidden,
            null,
            if (commit_dspark) (self.taps + 3 * b_usize * constants.hidden_dim) else null,
            null,
            self.blocks,
            block_count,
            w.output_query.?,
            w.final_norm,
            null,
            null,
            self.mlp_normed,
            null,
            null,
            null,
            null,
            null,
            1,
            batch_size,
            self.d_pos,
            last_mlp_abl.r,
            last_mlp_abl.d,
            last_mlp_abl.p,
            self.stream,
        );

        // Overlap dspark_commit on aux_stream concurrently with lm_head_argmax on stream!
        if (compute_lm_head) {
            if (commit_dspark) {
                tiger.assert_cuda(cuda.squid_cuda_event_record(self.fork_event, self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_wait_event(self.aux_stream, self.fork_event));
                cuda.squid_launch_dspark_commit(
                    &w.dspark,
                    self.taps,
                    &self.ds_k_cache,
                    &self.ds_v_cache,
                    &self.ds_k_scale,
                    &self.ds_v_scale,
                    &self.ds_kv_cap,
                    @intFromEnum(self.kv_format),
                    self.ds_f_ring.?,
                    self.ds_mix,
                    self.ds_kv,
                    batch_size,
                    self.d_pos,
                    self.aux_stream,
                );
                tiger.assert_cuda(cuda.squid_cuda_event_record(self.join_event, self.aux_stream));
            }
            cuda.squid_launch_lm_head_argmax(
                &w.unembed,
                self.mlp_normed,
                self.d_next_tokens,
                self.lm_logits,
                self.lm_blk_val,
                self.lm_blk_idx,
                self.d_sample_cfg,
                self.d_pos,
                if (is_verify) self.d_drafts else null,
                if (is_verify) self.d_draft_probs else null,
                if (is_verify) self.d_draft_q_sel else null,
                if (is_verify) self.d_draft_is_delta else null,
                if (is_verify) w.dspark.inv_draft_vocab_ids else null,
                w.dspark.draft_vocab_size,
                batch_size,
                self.stream,
            );
            if (commit_dspark) {
                tiger.assert_cuda(cuda.squid_cuda_stream_wait_event(self.stream, self.join_event));
            }
        } else if (commit_dspark) {
            cuda.squid_launch_dspark_commit(
                &w.dspark,
                self.taps,
                &self.ds_k_cache,
                &self.ds_v_cache,
                &self.ds_k_scale,
                &self.ds_v_scale,
                &self.ds_kv_cap,
                @intFromEnum(self.kv_format),
                self.ds_f_ring.?,
                self.ds_mix,
                self.ds_kv,
                batch_size,
                self.d_pos,
                self.stream,
            );
        }
    }

    fn ensure_b1_graph(self: *Engine) void {
        if (self.b1_graph_exec != null or !self.use_graphs) return;
        tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
        self.record_target_forward(1, false, true, false);
        cuda.squid_launch_inc_pos_and_copy_tok(self.d_pos, &self.d_tokens[0], &self.d_next_tokens[0], self.stream);
        tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.b1_graph_exec));
        tiger.assert(self.b1_graph_exec != null);
    }

    fn ensure_prefill_b8_graph(self: *Engine, commit_dspark: bool) void {
        const slot: usize = if (commit_dspark) 1 else 0;
        if (self.prefill_b8_graph_exec[slot] != null or !self.use_graphs) return;
        tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
        self.record_target_forward(constants.mtp_block_size_max, commit_dspark, false, false);
        cuda.squid_launch_inc_pos_by(self.d_pos, constants.mtp_block_size_max, self.stream);
        tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.prefill_b8_graph_exec[slot]));
        tiger.assert(self.prefill_b8_graph_exec[slot] != null);
    }

    fn ensure_spec_graph(self: *Engine, num_drafts: u32) void {
        tiger.assert(num_drafts >= 1 and num_drafts <= constants.mtp_draft_count_max);
        if (self.spec_graph_exec[num_drafts] != null or !self.use_graphs) return;
        const w = self.weights.?;
        const batch: i32 = @intCast(num_drafts + 1);
        const ds_ks_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_k_scale[0], self.ds_k_scale[1], self.ds_k_scale[2] };
        const ds_vs_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_v_scale[0], self.ds_v_scale[1], self.ds_v_scale[2] };

        tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
        cuda.squid_launch_dspark_draft(
            w.embed.?,
            w.embed_norm.?,
            &w.dspark,
            self.d_anchor,
            self.d_pos,
            @intCast(num_drafts),
            &self.ds_k_cache,
            &self.ds_v_cache,
            &ds_ks_const,
            &ds_vs_const,
            &self.ds_kv_cap,
            @intFromEnum(self.kv_format),
            self.split_buf,
            self.ds_f_ring.?,
            self.ds_ws,
            self.d_drafts,
            null,
            self.d_sample_cfg,
            self.d_draft_probs,
            self.d_draft_q_sel,
            self.d_draft_is_delta,
            self.stream,
        );
        cuda.squid_launch_assemble_verify_tokens(self.d_anchor, self.d_drafts, self.d_tokens, @intCast(num_drafts), self.stream);
        self.record_target_forward(batch, true, true, true);
        cuda.squid_launch_spec_verify(self.d_next_tokens, self.d_drafts, batch, self.d_accepted, self.d_num_acc, self.d_pos, self.d_anchor, self.stream);
        tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.spec_graph_exec[num_drafts]));
        tiger.assert(self.spec_graph_exec[num_drafts] != null);
    }

    fn ensure_adaptive_graphs(self: *Engine) void {
        if (!self.use_graphs) return;
        const w = self.weights.?;
        const ds_ks_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_k_scale[0], self.ds_k_scale[1], self.ds_k_scale[2] };
        const ds_vs_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_v_scale[0], self.ds_v_scale[1], self.ds_v_scale[2] };

        if (self.draft_graph_exec == null) {
            tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
            cuda.squid_launch_dspark_draft(
                w.embed.?,
                w.embed_norm.?,
                &w.dspark,
                self.d_anchor,
                self.d_pos,
                constants.mtp_draft_count_max,
                &self.ds_k_cache,
                &self.ds_v_cache,
                &ds_ks_const,
                &ds_vs_const,
                &self.ds_kv_cap,
                @intFromEnum(self.kv_format),
                self.split_buf,
                self.ds_f_ring.?,
                self.ds_ws,
                self.d_drafts,
                self.d_draft_conf,
                self.d_sample_cfg,
                self.d_draft_probs,
                self.d_draft_q_sel,
                self.d_draft_is_delta,
                self.stream,
            );
            tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.draft_graph_exec));
            tiger.assert(self.draft_graph_exec != null);
        }

        if (self.verify_b8_graph_exec == null) {
            tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
            cuda.squid_launch_assemble_verify_tokens(self.d_anchor, self.d_drafts, self.d_tokens, constants.mtp_draft_count_max, self.stream);
            self.record_target_forward(constants.mtp_block_size_max, true, true, true);
            cuda.squid_launch_spec_verify(self.d_next_tokens, self.d_drafts, constants.mtp_block_size_max, self.d_accepted, self.d_num_acc, self.d_pos, self.d_anchor, self.stream);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_num_accepted.?), @ptrCast(self.d_num_acc), @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_accepted.?), @ptrCast(self.d_accepted), constants.mtp_block_size_max * @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.verify_b8_graph_exec));
            tiger.assert(self.verify_b8_graph_exec != null);
        }

        if (self.adaptive_graph != null and self.adaptive_graph_threshold == self.spec_threshold) return;
        if (self.adaptive_graph) |old| {
            tiger.assert_cuda(cuda.squid_adaptive_graph_destroy(old));
            self.adaptive_graph = null;
        }

        tiger.assert_cuda(cuda.squid_adaptive_graph_create(
            self.d_anchor,
            self.d_drafts,
            self.d_draft_conf,
            self.spec_threshold,
            self.d_tokens,
            self.d_chosen_width,
            self.d_accepted,
            self.h_accepted.?,
            self.d_num_acc,
            self.h_num_accepted.?,
            self.h_chosen_width.?,
            self.stream,
            &self.adaptive_graph,
        ));
        const widths = [_]i32{ 2, 4, 8 };
        for (widths, 0..) |width, bidx| {
            tiger.assert_cuda(cuda.squid_adaptive_graph_branch_begin(self.adaptive_graph, @intCast(bidx), self.stream));
            self.record_target_forward(width, true, true, true);
            cuda.squid_launch_spec_verify(self.d_next_tokens, self.d_drafts, width, self.d_accepted, self.d_num_acc, self.d_pos, self.d_anchor, self.stream);
            tiger.assert_cuda(cuda.squid_adaptive_graph_branch_end(self.stream));
        }
        tiger.assert_cuda(cuda.squid_adaptive_graph_instantiate(self.adaptive_graph));
        self.adaptive_graph_threshold = self.spec_threshold;
    }
};

fn is_shared_loop_global_visit(vi: u32) bool {
    return vi == 11 or vi == 23 or vi == 17 or vi == 29;
}

fn shared_bulk_visit_idx(vi: u32) u32 {
    return switch (vi) {
        11, 23 => 35,
        17, 29 => 41,
        else => vi,
    };
}

fn visit_kv_cap(
    schedule: *const [constants.visit_count]constants.VisitInfo,
    vi: u32,
    context_tokens_max: u32,
    share_loop_kv: bool,
) u32 {
    if (!schedule[vi].is_global) return constants.swa_window_tokens + prefill_local_attn_sub;
    if (share_loop_kv and is_shared_loop_global_visit(vi)) return @min(context_tokens_max, constants.swa_window_tokens + 64);
    return context_tokens_max;
}

fn compute_runtime_vram_bytes(
    schedule: *const [constants.visit_count]constants.VisitInfo,
    context_tokens_max: u32,
    kv_format: constants.KVFormat,
    share_loop_kv: bool,
    has_dspark: bool,
    expert_groups_bytes: u64,
) u64 {
    const a = constants.vram_alignment_bytes;
    var total: u64 = 0;

    var vi: u32 = 0;
    while (vi < constants.visit_count) : (vi += 1) {
        const is_glob = schedule[vi].is_global;
        const cap: u32 = visit_kv_cap(schedule, vi, context_tokens_max, share_loop_kv);
        const vfmt: constants.KVFormat = if (is_glob) kv_format else .bf16;
        total += constants.kv_cache_pair_vram_bytes(cap, vfmt);
        total += 2 * tiger.align_up_u64(constants.sconv_ring_slots * constants.kv_dim * 4, a);
        total += 2 * tiger.align_up_u64(constants.sconv_ring_slots * constants.hidden_dim * 4, a);
    }

    if (has_dspark) {
        var ml: u32 = 0;
        while (ml < constants.mtp_layer_count) : (ml += 1) {
            const is_loc = constants.is_mtp_local_layer(ml);
            const cap: u32 = if (is_loc)
                (constants.swa_window_tokens + constants.mtp_block_size_max)
            else if (context_tokens_max > 524_288)
                65_536
            else
                context_tokens_max;
            const lfmt: constants.KVFormat = if (is_loc) .bf16 else kv_format;
            total += constants.kv_cache_pair_vram_bytes(cap, lfmt);
        }
        total += tiger.align_up_u64(constants.sconv_ring_slots * constants.hidden_dim * 4, a);
    }

    total += tiger.align_up_u64(constants.split_buffer_floats * 4, a);

    const b: u64 = constants.mtp_block_size_max;
    total += tiger.align_up_u64(4, a); // d_pos
    total += tiger.align_up_u64(b * 4, a); // d_tokens
    total += tiger.align_up_u64(b * 4, a); // d_next_tokens
    total += tiger.align_up_u64(4, a); // d_anchor
    total += tiger.align_up_u64(constants.mtp_draft_count_max * 4, a); // d_drafts
    total += tiger.align_up_u64(constants.mtp_draft_count_max * 4, a); // d_draft_conf
    total += tiger.align_up_u64(4, a); // d_chosen_width
    total += tiger.align_up_u64(b * 4, a); // d_accepted
    total += tiger.align_up_u64(4, a); // d_num_acc

    total += 3 * tiger.align_up_u64(b * constants.hidden_dim * 4, a); // hidden, prelude, partial
    total += tiger.align_up_u64(b * 4 * constants.hidden_dim * 4, a); // blocks
    total += 2 * tiger.align_up_u64(b * constants.hidden_dim * 4, a); // saved, prior_last
    total += 3 * (tiger.align_up_u64(b * constants.hidden_dim * 4, a) + tiger.align_up_u64(b * constants.hidden_dim * 2, a));

    total += tiger.align_up_u64(b * constants.qrgs_out_dim * 4, a);
    total += 2 * tiger.align_up_u64(b * constants.kv_dim * 4, a);
    total += tiger.align_up_u64(b * constants.query_dim * 4, a);
    total += tiger.align_up_u64(b * constants.query_dim * 2, a);
    total += 4 * tiger.align_up_u64(b * constants.hidden_dim * 4, a); // o_proj, residual, mlp_in, mlp_normed
    total += tiger.align_up_u64(b * constants.hidden_dim * 2, a); // mlp_normed_h
    total += tiger.align_up_u64(3 * b * constants.hidden_dim * 4, a); // ctrl_msg
    total += tiger.align_up_u64(3 * b * 2 * constants.hidden_dim * 4, a); // ctrl_cat
    total += tiger.align_up_u64(3 * b * constants.gate_rank * 4, a); // ctrl_d
    total += tiger.align_up_u64(b * constants.expert_count * 4, a); // router_logits
    total += 2 * tiger.align_up_u64(b * constants.top_k_experts * 4, a); // route_ids, route_w
    total += tiger.align_up_u64(expert_groups_bytes, a);
    total += tiger.align_up_u64(b * constants.top_k_experts * constants.intermediate_dim * 2, a); // moe_act_h
    total += tiger.align_up_u64(b * constants.hidden_dim * 4, a); // moe_down
    total += tiger.align_up_u64(b * constants.top_k_experts * constants.hidden_dim * 4, a); // moe_slot_out
    total += tiger.align_up_u64(4 * b * constants.hidden_dim * 4, a); // taps
    total += tiger.align_up_u64(3 * b * constants.hidden_dim * 4, a); // ds_mix
    total += tiger.align_up_u64(b * 2 * constants.kv_dim * 4, a); // ds_kv
    total += tiger.align_up_u64(constants.dspark_workspace_floats * 4, a); // ds_ws
    total += 2 * tiger.align_up_u64(b * 1024 * 4, a); // lm_blk_val, lm_blk_idx
    total += tiger.align_up_u64(b * constants.vocab_unpadded_count * 4, a); // lm_logits
    total += tiger.align_up_u64(constants.mtp_draft_count_max * constants.vocab_unpadded_count * 4, a); // d_draft_probs
    total += tiger.align_up_u64(constants.mtp_draft_count_max * 4, a); // d_draft_q_sel
    total += tiger.align_up_u64(4, a); // d_draft_is_delta
    total += tiger.align_up_u64(@sizeOf(cuda.CSampleConfig), a); // d_sample_cfg

    // Fast batched prefill arena (B <= 512)
    const pb: u64 = prefill_batch_max;
    total += tiger.align_up_u64(pb * 4, a); // p_tokens
    total += tiger.align_up_u64(prefill_sub_chunks_max * 4, a); // p_pos_vec
    total += 3 * tiger.align_up_u64(pb * constants.hidden_dim * 4, a); // p_hidden, p_prelude, p_partial
    total += tiger.align_up_u64(pb * 4 * constants.hidden_dim * 4, a); // p_blocks
    total += 2 * tiger.align_up_u64(pb * constants.hidden_dim * 4, a); // p_saved, p_prior_last
    total += 3 * (tiger.align_up_u64(pb * constants.hidden_dim * 4, a) + tiger.align_up_u64(pb * constants.hidden_dim * 2, a));
    total += tiger.align_up_u64(pb * constants.qrgs_out_dim * 4, a);
    total += 2 * tiger.align_up_u64(pb * constants.kv_dim * 4, a);
    total += tiger.align_up_u64(pb * constants.query_dim * 2, a);
    total += 4 * tiger.align_up_u64(pb * constants.hidden_dim * 4, a); // p_o_proj, p_residual, p_mlp_in, p_mlp_normed
    total += tiger.align_up_u64(pb * constants.hidden_dim * 2, a); // p_mlp_normed_h
    total += tiger.align_up_u64(pb * constants.expert_count * 4, a); // p_router_logits
    total += 2 * tiger.align_up_u64(pb * constants.top_k_experts * 4, a); // p_route_ids, p_route_w
    total += tiger.align_up_u64(1024 * 4, a); // p_exp_offsets + flat MoE tile table
    total += 3 * tiger.align_up_u64(pb * constants.top_k_experts * 4, a); // p_sorted_tok, p_sorted_slot, p_sorted_w
    total += tiger.align_up_u64(pb * constants.top_k_experts * constants.intermediate_dim * 2, a); // p_moe_act_h
    total += tiger.align_up_u64(pb * constants.top_k_experts * constants.hidden_dim * 4, a); // p_moe_slot_out
    total += tiger.align_up_u64(4 * pb * constants.hidden_dim * 4, a); // p_taps
    total += tiger.align_up_u64(@as(u64, constants.qrgs_out_dim) * constants.hidden_dim * 2, a); // p_w_f16
    total += tiger.align_up_u64(pb * constants.hidden_dim * 4, a); // p_ctrl_msg
    total += tiger.align_up_u64(pb * 2 * constants.hidden_dim * 2, a); // p_ctrl_cat_h
    total += tiger.align_up_u64(pb * constants.gate_rank * 4, a); // p_ctrl_d
    total += tiger.align_up_u64(pb * constants.gate_rank * 2, a); // p_ctrl_d_h

    return total;
}

const AblationSlice = struct {
    r: ?[*]const f32 = null,
    d: ?[*]const f32 = null,
    p: ?[*]const f32 = null,
};

fn ablation_slice(w: *const weights_mod.ModelWeights, is_mlp: bool, visit_idx: u32) AblationSlice {
    if (!w.ablation.active) return .{};
    const v: usize = @intCast(visit_idx);
    const active = if (is_mlp) w.ablation.mlp_nonzero[v] else w.ablation.attn_nonzero[v];
    if (!active) return .{};
    const dirs = if (is_mlp) w.ablation.mlp_dirs.? else w.ablation.attn_dirs.?;
    const det_opt = if (is_mlp) w.ablation.mlp_det else w.ablation.attn_det;
    const params = if (is_mlp) w.ablation.mlp_params.? else w.ablation.attn_params.?;
    return .{
        .r = dirs + v * constants.hidden_dim,
        .d = if (det_opt) |det| det + v * constants.hidden_dim else null,
        .p = params + v * 4,
    };
}

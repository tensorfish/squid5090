#pragma once

#include "config.hpp"
#include "weights.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace squid {

// Maximum verify batch size supported by fast multi-row kernels
constexpr int MAX_VERIFY_B = 8;

struct ExpertGroups {
    int num_unique;
    int expert_id[MAX_VERIFY_B * TOP_K];
    uint8_t token_mask[MAX_VERIFY_B * TOP_K];
    uint8_t slot_idx[MAX_VERIFY_B * TOP_K][MAX_VERIFY_B];
    float route_w[MAX_VERIFY_B * TOP_K][MAX_VERIFY_B];
};

// --- GEMV / Small-Batch Linear & MoE Kernels (kernels_gemv.cu) ---

// Y[B, N] = X[B, K] @ W^T  (B = 1..MAX_VERIFY_B, or general B via chunking)
// If add_residual == true, Y += X @ W^T
void launch_linear(
    const LinearWeight& w,
    const float* x, // [B, K]
    float* y,       // [B, N]
    int B,
    bool add_residual,
    cudaStream_t stream);

// Raw BF16 matrix multiply Y[B, N] = X[B, K] @ W[N, K]^T (for small dense matrices like controllers/sconv_proj)
void launch_bf16_linear(
    const __nv_bfloat16* w,
    const float* x,
    float* y,
    int B, int N, int K,
    cudaStream_t stream);

// Fused LM Head + Softcap(30) + GPU Argmax / Leviathan-Chen Stochastic Rejection Sampling!
// sample_cfg->inv_temp == 0: pure greedy argmax (zero logits write).
// sample_cfg->inv_temp <  0: write post-softcap logits for KLD eval + greedy argmax.
// sample_cfg->inv_temp >  0: temperature/top_p/top_k sampling (with exact rejection sampling when d_draft_tokens != nullptr).
void launch_lm_head_argmax(
    const LinearWeight& unembed,
    const float* x,                      // [B, HIDDEN]
    int32_t* out_tokens,                 // [B]
    float* out_logits,                   // optional [B, VOCAB_UNPADDED]
    float* scratch_vals,                 // [B, 1024]
    int32_t* scratch_idxs,               // [B, 1024]
    const SampleConfig* sample_cfg,      // optional device pointer
    const int32_t* d_pos,                // optional device pointer (batch start pos)
    const int32_t* d_draft_tokens,       // optional [B-1] for rejection sampling
    const float* d_draft_probs,          // optional [B-1, draft_vocab_size] normalized q_b(v)
    const float* d_draft_q_sel,          // optional [B-1] q_b(draft_tokens[b])
    const int32_t* d_draft_is_delta,     // optional [1] non-zero when prompt-lookup (q = delta_x)
    const int32_t* inv_draft_vocab,      // optional [VOCAB_UNPADDED] real->draft idx (-1 if absent)
    int draft_vocab_size,
    int B,
    cudaStream_t stream);

// Build deduplicated expert groups for B tokens (B = 2..8)
void launch_build_expert_groups(
    const int32_t* route_ids,  // [B, TOP_K]
    const float* route_w,      // [B, TOP_K]
    ExpertGroups* groups,
    int B,
    cudaStream_t stream);

// Fast zero-SMEM L1-cached half-activation linear projections!
void launch_linear_h(
    const LinearWeight& w,
    const __half* x_h, // [B, K]
    float* y,          // [B, N]
    int B,
    bool add_residual,
    cudaStream_t stream);

// Fused Q+R+Gate+Sink (6552) + K (1024) + V (1024) in ONE single kernel launch!
void launch_qkvrg_fused(
    const LayerWeights& lw,
    const __half* in_q_h,
    const __half* in_k_h,
    const __half* in_v_h,
    float* qrgs_out,  // [B, 6552]
    float* k_out,     // [B, 1024]
    float* v_out,     // [B, 1024]
    int B,
    cudaStream_t stream);

void launch_f32_to_f16(const float* src, __half* dst, int n, cudaStream_t stream);

// MoE Gate+Up + Softcapped SwiGLU (reads half x, writes half out_act!)
void launch_moe_gate_up_h(
    const ExpertBank& w13,
    const __half* x_h,         // [B, HIDDEN]
    const int32_t* route_ids,  // [B, TOP_K]
    const ExpertGroups* groups,// used when B > 1
    __half* out_act_h,         // [B, TOP_K, INTERMEDIATE]
    int B,
    cudaStream_t stream);

// MoE Down + Weighted Top-K Reduction (reads half act!)
void launch_moe_down_h(
    const ExpertBank& w2,
    const __half* act_h,       // [B, TOP_K, INTERMEDIATE]
    const int32_t* route_ids,  // [B, TOP_K]
    const float* route_w,      // [B, TOP_K]
    const ExpertGroups* groups,// used when B > 1
    float* out_down,           // [B, HIDDEN]
    int B,
    cudaStream_t stream,
    bool already_zeroed = false);

// --- Inkling-2-Mini Specialized Ops (kernels_inkling.cu) ---

// Embed lookup + RMSNorm(embed_norm)
void launch_embed_norm(
    const __nv_bfloat16* embed,
    const __nv_bfloat16* embed_norm,
    const int32_t* tokens, // [B]
    float* out,            // [B, HIDDEN]
    int B,
    cudaStream_t stream);

// RMSNorm: out[b, g, d] = x[b, g, d] / sqrt(mean(x^2) + eps) * (weight ? weight[d] : 1)
void launch_rmsnorm(
    const float* x,
    const __nv_bfloat16* weight, // optional (nullptr = no gamma)
    float* out,
    int B, int width, int group_size, float eps,
    cudaStream_t stream);

// Depth attention over [blocks[0..n_blocks-1], partial] with query[q_idx]
// blocks has stride_b = max_blocks * HIDDEN per batch row (or separate pointers? layout: [B, max_blocks, HIDDEN])
void launch_depth_attn(
    const float* blocks,   // [B, 4, HIDDEN] (or [B, 1, HIDDEN] for saved)
    int max_blocks,
    int n_blocks,
    const float* partial,  // [B, HIDDEN]
    const float* query,    // [n_queries, HIDDEN] F32
    int q_idx,
    float* out,            // [B, HIDDEN]
    int B,
    cudaStream_t stream);

// Fused partial = rmsnorm(hidden, null) + depth_attn + optional rmsnorm(post_norm_w)
// Handles n_out = 1 (local) or n_out = 3 (global Q/K/V) in ONE single kernel!
// Also writes half copies into out_h0/1/2 if non-null (zero extra kernel!).
void launch_fused_depth_norm(
    const float* hidden,         // [B, HIDDEN]
    float* partial_out,          // optional [B, HIDDEN] (written if non-null)
    const float* blocks,         // [B, 4, HIDDEN]
    int n_blocks,
    const float* queries,        // [n_out, HIDDEN] F32 (nullptr when l==0 -> skip depth_attn)
    const __nv_bfloat16* norm0,  // optional
    const __nv_bfloat16* norm1,  // optional
    const __nv_bfloat16* norm2,  // optional
    float* out0,                 // [B, HIDDEN]
    float* out1,                 // optional [B, HIDDEN]
    float* out2,                 // optional [B, HIDDEN]
    __half* out_h0,              // optional [B, HIDDEN]
    __half* out_h1,              // optional [B, HIDDEN]
    __half* out_h2,              // optional [B, HIDDEN]
    int n_out,
    int B,
    cudaStream_t stream);

// Batched parallel Cross-Pass Controller for 1 stream (MLP) or 3 streams (Attn Q/K/V)!
void launch_controller_merge_batched(
    float* const* xs,                   // [n_streams] each [B, HIDDEN]
    const float* saved,                 // [B, 1, HIDDEN]
    const float* prior_last,            // [B, HIDDEN]
    const float* cp_queries[3],         // [n_streams] each [HIDDEN] F32
    const ControllerWeights* cws[3],    // [n_streams]
    const __nv_bfloat16* post_norms[3], // [n_streams]
    float* const* outs,                 // [n_streams] each [B, HIDDEN]
    __half* const* outs_h,              // optional [n_streams] each [B, HIDDEN]
    int n_streams,
    float* scratch_msg,                 // [3 * B * HIDDEN]
    float* scratch_cat,                 // [3 * B * 2 * HIDDEN]
    float* scratch_d,                   // [3 * B * GATE_RANK]
    int B,
    cudaStream_t stream);

// Static Short Conv (with ring buffer update):
// Stores x[b] into ring[(pos+b) % 4], computes (sum_t w[c,t]*ring[pos+b-3+t, c] + x[b,c]) * inv_sqrt_5
void launch_static_sconv(
    const float* x,             // [B, width]
    const __nv_bfloat16* w,     // [width, 4]
    float* ring,                // [4, width]
    float* out,                 // [B, width]
    int B, int width, const int32_t* d_pos,
    cudaStream_t stream);

// Dynamic Short Conv (for global layer K & V):
// First computes dynamic taps = weight_proj @ dyn_in ([B, 32]), updates ring, computes:
// (sum_t ring[pos+b-t, c] * (static_w[t, c] + dyn_taps[b, head*4 + t]) + x[b, c]) / 3.0f
void launch_dynamic_sconv(
    const float* x,             // [B, KV_DIM]
    const float* dyn_in,        // [B, HIDDEN]
    const __nv_bfloat16* static_w,    // [4, KV_DIM]
    const __nv_bfloat16* weight_proj, // [32, HIDDEN]
    float* ring,                // [4, KV_DIM]
    float* out,                 // [B, KV_DIM]
    float* scratch_taps,        // [B, 32]
    int B, const int32_t* d_pos,
    cudaStream_t stream);

// Fused Post-Proj Prep for Attention:
// - V: RMSNorm(head_dim=128, no gamma) -> store to V_cache[slot] (BF16)
// - K: L2-clip(128, QK_CLIP, no scale) -> store to K_cache[slot] (BF16)
// - Q: L2-clip(128, QK_CLIP, q_clip_scale) + optional global log-pos scale
// - R: L2-clip(16, REL_CLIP, rel_clip_scale) + optional global log-pos scale
void launch_attn_prep_store(
    float* qrgs,                // [B, 6552] (Q at 0, R at 3072, Gate at 3456, Sink at 6528)
    const float* ck,            // [B, KV_DIM]
    const float* cv,            // [B, KV_DIM]
    const __nv_bfloat16* q_clip_scale,
    const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache,     // [cap, KV_DIM]
    __nv_bfloat16* v_cache,     // [cap, KV_DIM]
    int kv_cap,
    bool is_global,
    int B, const int32_t* d_pos,
    cudaStream_t stream);

// Relative Bias + Attention with Sink + Gated Output (out * sigmoid(gate)):
// For local visits (window > 0): runs inline relative-bias FlashDecoding on BF16 SWA KV cache.
// For global visits (window == 0): runs 192-CTA GQA*B-shared Split-K FlashDecoding over bulk [0 .. tail_start)
// in BF16/FP8/Q4 KV format, then fuses tail [tail_start .. pos+b] with exact relative bias & online merge!
void launch_relative_attention_gated(
    const float* qrgs,                  // [B, 6552]
    const __nv_bfloat16* rel_proj,      // [16, 1024]
    const void* k_cache,                // [cap, KV_DIM] (BF16/FP8) or [cap, KV_DIM/2] (Q4)
    const void* v_cache,                // [cap, KV_DIM] (BF16/FP8) or [cap, KV_DIM/2] (Q4)
    const __half* k_scale,              // optional: FP8 [cap, 8], Q4 [cap, 16]
    const __half* v_scale,              // optional: FP8 [cap, 8], Q4 [cap, 16]
    const void* k_bulk_cache,           // optional shared last-loop bulk KV for [0 .. pos-1023)
    const void* v_bulk_cache,
    const __half* k_bulk_scale,
    const __half* v_bulk_scale,
    int bulk_kv_cap,
    KVFormat kv_fmt,
    float* split_buf,                   // [600000] floats scratch for global Split-K partials
    float* out_gated,                   // [B, 3072]
    __half* out_gated_h,                // optional [B, 3072]
    int kv_cap,
    int window,                         // 1024 for local, 0 for global
    int B, const int32_t* d_pos,
    cudaStream_t stream);

// 64-block parallel half-warp NVFP4 MoE Router + Fused Top-8 & ExpertGroups + Free moe_down zeroing!
void launch_route_topk_nvfp4(
    const __half* m_h,            // [B, HIDDEN]
    const LinearWeight& router_nvfp4,
    const float* router_bias,     // [128]
    float* scratch_logits,        // [B, 128]
    int32_t* out_ids,             // [B, TOP_K]
    float* out_weights,           // [B, TOP_K]
    ExpertGroups* groups,         // optional (filled when B > 1)
    float* moe_down_zero,         // optional (zeroed in parallel when B > 1!)
    int B,
    cudaStream_t stream);

// Fused MoE Router (BF16): parallel 64-block GEMV [128, 2304] + Fused Top-8 & ExpertGroups!
void launch_route_topk(
    const float* m,               // [B, HIDDEN]
    const __nv_bfloat16* router_w,// [128, HIDDEN]
    const float* router_bias,     // [128]
    float* scratch_logits,        // [B, 128]
    int32_t* out_ids,             // [B, TOP_K]
    float* out_weights,           // [B, TOP_K]
    ExpertGroups* groups,         // optional (filled when B > 1)
    float* moe_down_zero,         // optional (zeroed in parallel when B > 1!)
    int B,
    cudaStream_t stream);

// Multi-SM Batched NVFP4 Cross-Pass Controller (shares weights across B tokens across 32-108 SMs!)
void launch_controller_nvfp4_batched(
    float* const* xs,
    const float* saved,
    const float* prior_last,
    const float* cp_queries[3],
    const ControllerWeights* cws[3],
    const __nv_bfloat16* post_norms[3],
    float* const* outs,
    __half* const* outs_h,
    int n_streams,
    float* scratch_msg,
    __half* scratch_cat_h,
    __half* scratch_d_h,
    int B,
    cudaStream_t stream);

// MEGA-FUSION 1: Single-kernel NVFP4 cross-pass controller (replaces 4 kernels!)
void launch_controller_mega_nvfp4(
    float* const* xs,
    const float* saved,
    const float* prior_last,
    const float* cp_queries[3],
    const ControllerWeights* cws[3],
    const __nv_bfloat16* post_norms[3],
    float* const* outs,
    __half* const* outs_h,
    int n_streams,
    int B,
    cudaStream_t stream);

// MEGA-FUSION 6: Global-layer fused dynamic k_sconv + v_sconv + Q/R/K clip + log-pos scale + V norm + KV store (BF16/FP8/Q4!)
void launch_mega_global_sconv_prep_store(
    float* qrgs,
    const float* k_proj,
    const float* v_proj,
    const float* in_k,
    const float* in_v,
    const __nv_bfloat16* k_sconv_w,
    const __nv_bfloat16* v_sconv_w,
    const __nv_bfloat16* k_sconv_proj,
    const __nv_bfloat16* v_sconv_proj,
    float* k_ring,
    float* v_ring,
    const __nv_bfloat16* q_clip_scale,
    const __nv_bfloat16* rel_clip_scale,
    void* k_cache,
    void* v_cache,
    __half* k_scale,
    __half* v_scale,
    KVFormat kv_fmt,
    int kv_cap,
    int B, const int32_t* d_pos,
    cudaStream_t stream);

// MEGA-FUSION 5: Fuses mlp_post_res(vi-1) [with optional deterministic 8-slot reduction!] + pre_tap + next attn_in_depth_norm(vi) (+ post_tap)!
void launch_mega_mlp_post_next_depth_norm(
    const float* moe_down_or_slots,
    bool moe_is_slotted,
    const __nv_bfloat16* prev_mlp_sconv_w,
    float* prev_mlp_ring,
    const float* residual,
    float* hidden_out,           // [B, HIDDEN]
    float* pre_tap_out,          // optional [B, HIDDEN] (taps 0/1/2 or prelude_)
    float* post_tap_out,         // optional [B, HIDDEN] (final tap 3)
    float* partial_out,          // optional [B, HIDDEN]
    const float* blocks,         // [B, 4, HIDDEN]
    int n_blocks,
    const float* queries,        // [n_out, HIDDEN] F32
    const __nv_bfloat16* norm0,
    const __nv_bfloat16* norm1,
    const __nv_bfloat16* norm2,
    float* out0,
    float* out1,
    float* out2,
    __half* out_h0,
    __half* out_h1,
    __half* out_h2,
    int n_out,
    int B,
    const int32_t* d_pos,
    const float* abl_r,
    const float* abl_d,
    const float* abl_p,
    cudaStream_t stream);

// MEGA-FUSION 2: Local-layer fused k_sconv + v_sconv + Q/R/K clip + V norm + KV cache store (replaces 3 kernels!)
void launch_mega_local_sconv_prep_store(
    float* qrgs,
    const float* k_proj,
    const float* v_proj,
    const __nv_bfloat16* k_sconv_w,
    const __nv_bfloat16* v_sconv_w,
    float* k_ring,
    float* v_ring,
    const __nv_bfloat16* q_clip_scale,
    const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int kv_cap,
    int B, const int32_t* d_pos,
    cudaStream_t stream);

// MEGA-FUSION 3: attn_sconv + residual/block-boundary + mlp_depth_attn + optional mlp_norm (+ half!) (replaces 4 kernels!)
void launch_mega_attn_post_mlp_in(
    const float* o_proj,
    const __nv_bfloat16* attn_sconv_w,
    float* attn_ring,
    const float* hidden,
    const float* partial_prev, // written to blocks[block_count] if is_boundary
    float* blocks,             // [B, 4, HIDDEN]
    int block_count_before,
    bool is_boundary,
    float* residual_out,       // [B, HIDDEN]
    const float* mlp_query,    // [HIDDEN] F32
    const __nv_bfloat16* mlp_norm_w, // nullptr if use_prior (controller runs next!)
    float* mlp_out,            // [B, HIDDEN]
    __half* mlp_out_h,         // optional [B, HIDDEN]
    int B, const int32_t* d_pos,
    const float* abl_r,
    const float* abl_d,
    const float* abl_p,
    cudaStream_t stream);

// MEGA-FUSION 4: mlp_sconv + residual add -> hidden (supports deterministic 8-slot reduction when B>1!)
void launch_mega_mlp_post_res(
    const float* moe_down_or_slots,
    bool moe_is_slotted,
    const __nv_bfloat16* mlp_sconv_w,
    float* mlp_ring,
    const float* residual,
    float* hidden_out,
    int B, const int32_t* d_pos,
    const float* abl_r,
    const float* abl_d,
    const float* abl_p,
    cudaStream_t stream);

// Simple vector ops
void launch_vec_add(const float* a, const float* b, float* out, int n, cudaStream_t stream);
void launch_copy_block(const float* src, float* blocks, int max_blocks, int slot_idx, int B, cudaStream_t stream);
void launch_extract_block(const float* blocks, int max_blocks, int slot_idx, float* dst, int B, cudaStream_t stream);

// --- DSpark MTP Speculative Decoding Kernels (kernels_dspark.cu) ---

// Commit target taps [B, 4, HIDDEN] for positions d_pos..d_pos+B-1 into DSpark context KV caches & markov f-ring
void launch_dspark_commit(
    const DsparkWeights& dw,
    const float* taps,          // [4, B, HIDDEN] (4 tap slots, each [B, HIDDEN])
    void* ctx_k[MTP_LAYERS],    // local BF16, global BF16/FP8/Q4
    void* ctx_v[MTP_LAYERS],
    __half* ctx_ks[MTP_LAYERS], // optional scales for global layer 1
    __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS],
    KVFormat global_kv_fmt,
    float* f_ring,              // [16, HIDDEN] ring buffer of markov context f
    float* scratch_mix,         // [4, B, HIDDEN]
    float* scratch_kv,          // [B, 2*KV_DIM]
    int B, const int32_t* d_pos,
    cudaStream_t stream);

// Run DSpark 3-layer block forward (anchor + 7 masks = 8 rows) + 7-step Markov RNN head on GPU!
// Writes draft_tokens[0..num_drafts-1] directly on device!
void launch_dspark_draft(
    const ModelWeights& mw,
    const int32_t* d_anchor_token, // [1] anchor token at position d_pos
    const int32_t* d_pos,          // [1] position of anchor token
    int num_drafts,                // 1..7
    void* ctx_k[MTP_LAYERS],
    void* ctx_v[MTP_LAYERS],
    const __half* ctx_ks[MTP_LAYERS],
    const __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS],
    KVFormat global_kv_fmt,
    float* split_buf,              // [600000] floats scratch for global Split-K
    const float* f_ring,           // [16, HIDDEN], reads (pos - 1) & 15
    float* workspace,              // preallocated workspace for DSpark block & Markov head
    int32_t* out_draft_tokens,     // [num_drafts]
    float* out_draft_conf,         // optional [num_drafts] max softmax probability
    const SampleConfig* sample_cfg,// optional device sampling config (inv_temp > 0 => stochastic draft!)
    float* out_draft_probs,        // optional [num_drafts, draft_vocab_size] normalized q_i(v)
    float* out_draft_q_sel,        // optional [num_drafts] q_i(out_draft_tokens[i])
    int32_t* out_draft_is_delta,   // optional [1] cleared to 0 by DSpark
    cudaStream_t stream);

// GPU verification kernel: compares target's predicted tokens [B] against draft tokens [B-1],
// computes n_accepted (1..B) and next anchor token, updates d_pos += n_accepted!
void launch_spec_verify(
    const int32_t* target_preds,   // [B] (target argmax at positions pos..pos+B-1)
    const int32_t* draft_tokens,   // [B-1] (proposed tokens at pos+1..pos+B-1)
    int B,
    int32_t* out_accepted_tokens,  // [B]
    int32_t* out_num_accepted,     // [1]
    int32_t* d_pos,                // [1] updated += num_accepted
    int32_t* d_next_anchor,        // [1] = accepted[num_accepted - 1]
    cudaStream_t stream);

// Fast Batched Prefill Kernels (B <= 256): cuBLAS FP16 TC GEMM + Blackwell TC MoE + Batched sconv/ctrl
void launch_prefill_warmup_cublas(cudaStream_t stream);

void launch_prefill_f32_to_f16(const float* src, __half* dst, int count, cudaStream_t stream);

void launch_prefill_gemm_h(
    const LinearWeight& w,
    const __half* x_h,
    float* y,
    __half* w_f16_scratch,
    int B,
    cudaStream_t stream);

void launch_prefill_sconv_post_res(
    const float* src_or_slots,
    int is_slotted,
    const __nv_bfloat16* sconv_w,
    float* ring,
    const float* residual,
    float* out_res,
    float* pre_tap_out,
    float* blocks,
    const float* partial,
    int block_idx,
    int is_boundary,
    int B,
    const int32_t* d_pos,
    const float* abl_r,
    const float* abl_d,
    const float* abl_p,
    cudaStream_t stream);

void launch_prefill_copy_taps(
    const float* src,
    float* dst_taps,
    int tap_slot,
    int B,
    cudaStream_t stream);

void launch_prefill_local_sconv_prep_store(
    float* qrgs,
    float* k_proj,
    float* v_proj,
    const __nv_bfloat16* k_sconv_w,
    const __nv_bfloat16* v_sconv_w,
    float* k_ring,
    float* v_ring,
    const __nv_bfloat16* q_clip_scale,
    const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int kv_cap,
    int B,
    const int32_t* d_pos,
    cudaStream_t stream);

void launch_prefill_global_sconv_prep_store(
    float* qrgs,
    float* k_proj,
    float* v_proj,
    const __half* in_k_h,
    const __half* in_v_h,
    const __nv_bfloat16* k_sconv_w,
    const __nv_bfloat16* v_sconv_w,
    const __nv_bfloat16* k_sconv_proj,
    const __nv_bfloat16* v_sconv_proj,
    float* k_ring,
    float* v_ring,
    const __nv_bfloat16* q_clip_scale,
    const __nv_bfloat16* rel_clip_scale,
    void* k_cache,
    void* v_cache,
    __half* k_scale,
    __half* v_scale,
    KVFormat kv_fmt,
    int kv_cap,
    float* scratch_taps,
    __half* w_f16_scratch,
    int B,
    const int32_t* d_pos,
    cudaStream_t stream);

void launch_prefill_flash_attn_tc(
    const float* qrgs,
    const __nv_bfloat16* rel_proj,
    const void* k_cache,
    const void* v_cache,
    const __half* k_scale,
    const __half* v_scale,
    const void* k_bulk_cache,
    const void* v_bulk_cache,
    const __half* k_bulk_scale,
    const __half* v_bulk_scale,
    int bulk_kv_cap,
    KVFormat kv_fmt,
    __half* out_gated_h,
    int kv_cap,
    int window,
    int B,
    const int32_t* d_pos,
    cudaStream_t stream);

void launch_prefill_controller_one(
    const float* x,
    const float* saved,
    const float* prior_last,
    const float* cp_q,
    const ControllerWeights& cw,
    const __nv_bfloat16* post_norm,
    float* out,
    __half* out_h,
    float* scratch_msg,
    __half* scratch_cat_h,
    float* scratch_d,
    __half* scratch_d_h,
    float* scratch_f,
    float* scratch_g,
    __half* w_f16_scratch,
    int B,
    cudaStream_t stream);

void launch_prefill_route_and_sort(
    const float* router_logits,
    const float* router_bias,
    int32_t* route_ids,
    float* route_w,
    int32_t* expert_offsets,
    int32_t* sorted_tok,
    int32_t* sorted_slot,
    float* sorted_w,
    int B,
    cudaStream_t stream);

void launch_prefill_moe_gate_up_h(
    const ExpertBank& w13,
    const __half* x_h,
    const int32_t* expert_offsets,
    const int32_t* sorted_tok,
    const int32_t* sorted_slot,
    __half* out_act_h,
    int B,
    cudaStream_t stream);

void launch_prefill_moe_down_h(
    const ExpertBank& w2,
    const __half* act_h,
    const int32_t* expert_offsets,
    const int32_t* sorted_tok,
    const int32_t* sorted_slot,
    const float* sorted_w,
    float* out_slots,
    int B,
    cudaStream_t stream);

} // namespace squid

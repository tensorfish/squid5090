// squid5090-zig — TigerStyle Zig + Blackwell sm_120a CUDA Inference Engine & OpenAI Server.
// Zero dynamic allocations post-init(), single-arena VRAM weights & runtime, always-on assertions.

const std = @import("std");
const constants = @import("constants.zig");
const engine_mod = @import("engine.zig");
const server_mod = @import("server.zig");
const tiger = @import("tiger.zig");
const tokenizer_mod = @import("tokenizer.zig");
const weights_mod = @import("weights.zig");

const bench_runs_max: usize = 64;
const cli_tokens_max: usize = 1_048_576;

pub fn main(init: std.process.Init.Minimal) u8 {
    const argv = init.args.vector;
    if (argv.len < 3) {
        print_usage();
        return 1;
    }

    const command = std.mem.span(argv[1]);
    const model_directory = std.mem.span(argv[2]);

    var tokens_csv: []const u8 = "1,2,3";
    var tokens_file: []const u8 = "";
    var prompt_text: []const u8 = "";
    var prompt_file: []const u8 = "";
    var mode_str: []const u8 = if (std.mem.eql(u8, command, "serve")) "fp8" else "exact";
    var kv_str: []const u8 = "auto";
    var host: []const u8 = "0.0.0.0";
    var model_name: []const u8 = "inkling-2-mini";
    var model_name_explicit: bool = false;
    var ablation_path: []const u8 = "";
    var out_path: []const u8 = "";
    var ref_path: []const u8 = "";
    var effort: []const u8 = "none";
    var api_key: []const u8 = "";
    if (tiger.getenv("SQUID_API_KEY")) |env_key| api_key = std.mem.span(env_key);

    var port: u16 = 8000;
    var mtp: u32 = if (std.mem.eql(u8, command, "serve")) 7 else 0;
    var spec_threshold: f32 = 0.80;
    var temperature: f32 = if (std.mem.eql(u8, command, "serve")) 1.0 else 0.0;
    var top_p: f32 = 0.95;
    var top_k: i32 = 0;
    var seed: u64 = 0;
    var max_tokens: u32 = 128;
    var runs: u32 = 6;
    var ctx: u32 = if (std.mem.eql(u8, command, "serve")) 65536 else 8192;
    var use_graphs: bool = true;
    var use_prompt_lookup: bool = true;
    var embed_on_host: bool = false;
    var share_loop_kv: bool = false;
    if (tiger.getenv("SQUID_SHARE_LOOP_KV")) |skv| {
        if (std.mem.eql(u8, std.mem.span(skv), "1")) share_loop_kv = true;
    }
    var raw_prompt: bool = false;
    var raw_thinking: bool = false;
    var keep_history_thinking: bool = false;
    var has_tokens_arg: bool = false;

    var arg_i: usize = 3;
    while (arg_i < argv.len) : (arg_i += 1) {
        const arg = std.mem.span(argv[arg_i]);
        if (std.mem.eql(u8, arg, "--tokens") and arg_i + 1 < argv.len) {
            arg_i += 1;
            tokens_csv = std.mem.span(argv[arg_i]);
            has_tokens_arg = true;
        } else if (std.mem.eql(u8, arg, "--tokens-file") and arg_i + 1 < argv.len) {
            arg_i += 1;
            tokens_file = std.mem.span(argv[arg_i]);
            has_tokens_arg = true;
        } else if (std.mem.eql(u8, arg, "--prompt") and arg_i + 1 < argv.len) {
            arg_i += 1;
            prompt_text = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--prompt-file") and arg_i + 1 < argv.len) {
            arg_i += 1;
            prompt_file = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--mode") and arg_i + 1 < argv.len) {
            arg_i += 1;
            mode_str = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--kv") and arg_i + 1 < argv.len) {
            arg_i += 1;
            kv_str = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--host") and arg_i + 1 < argv.len) {
            arg_i += 1;
            host = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--port") and arg_i + 1 < argv.len) {
            arg_i += 1;
            port = std.fmt.parseInt(u16, std.mem.span(argv[arg_i]), 10) catch 8000;
        } else if (std.mem.eql(u8, arg, "--model-name") and arg_i + 1 < argv.len) {
            arg_i += 1;
            model_name = std.mem.span(argv[arg_i]);
            model_name_explicit = true;
        } else if (std.mem.eql(u8, arg, "--ablation") and arg_i + 1 < argv.len) {
            arg_i += 1;
            ablation_path = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--out") and arg_i + 1 < argv.len) {
            arg_i += 1;
            out_path = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--ref") and arg_i + 1 < argv.len) {
            arg_i += 1;
            ref_path = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--effort") and arg_i + 1 < argv.len) {
            arg_i += 1;
            effort = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--api-key") and arg_i + 1 < argv.len) {
            arg_i += 1;
            api_key = std.mem.span(argv[arg_i]);
        } else if (std.mem.eql(u8, arg, "--mtp") and arg_i + 1 < argv.len) {
            arg_i += 1;
            mtp = std.fmt.parseInt(u32, std.mem.span(argv[arg_i]), 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--spec-threshold") and arg_i + 1 < argv.len) {
            arg_i += 1;
            const v = std.fmt.parseFloat(f32, std.mem.span(argv[arg_i])) catch 0.80;
            spec_threshold = if (std.math.isFinite(v)) v else 0.80;
        } else if (std.mem.eql(u8, arg, "--temp") and arg_i + 1 < argv.len) {
            arg_i += 1;
            const v = std.fmt.parseFloat(f32, std.mem.span(argv[arg_i])) catch 0.0;
            temperature = if (std.math.isFinite(v) and v >= 0.0) v else 0.0;
        } else if (std.mem.eql(u8, arg, "--top-p") and arg_i + 1 < argv.len) {
            arg_i += 1;
            const v = std.fmt.parseFloat(f32, std.mem.span(argv[arg_i])) catch 0.95;
            top_p = if (std.math.isFinite(v) and v > 0.0 and v <= 1.0) v else 0.95;
        } else if (std.mem.eql(u8, arg, "--top-k") and arg_i + 1 < argv.len) {
            arg_i += 1;
            top_k = std.fmt.parseInt(i32, std.mem.span(argv[arg_i]), 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--seed") and arg_i + 1 < argv.len) {
            arg_i += 1;
            seed = std.fmt.parseInt(u64, std.mem.span(argv[arg_i]), 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--max-tokens") and arg_i + 1 < argv.len) {
            arg_i += 1;
            max_tokens = std.fmt.parseInt(u32, std.mem.span(argv[arg_i]), 10) catch 128;
        } else if (std.mem.eql(u8, arg, "--runs") and arg_i + 1 < argv.len) {
            arg_i += 1;
            runs = std.fmt.parseInt(u32, std.mem.span(argv[arg_i]), 10) catch 6;
        } else if (std.mem.eql(u8, arg, "--ctx") and arg_i + 1 < argv.len) {
            arg_i += 1;
            ctx = std.fmt.parseInt(u32, std.mem.span(argv[arg_i]), 10) catch 8192;
        } else if (std.mem.eql(u8, arg, "--embed-host")) {
            embed_on_host = true;
        } else if (std.mem.eql(u8, arg, "--share-loop-kv")) {
            share_loop_kv = true;
        } else if (std.mem.eql(u8, arg, "--no-graph")) {
            use_graphs = false;
        } else if (std.mem.eql(u8, arg, "--no-prompt-lookup")) {
            use_prompt_lookup = false;
        } else if (std.mem.eql(u8, arg, "--raw")) {
            raw_prompt = true;
        } else if (std.mem.eql(u8, arg, "--raw-thinking")) {
            raw_thinking = true;
        } else if (std.mem.eql(u8, arg, "--keep-history-thinking")) {
            keep_history_thinking = true;
        }
    }

    const prompt_file_buf: ?[]u8 = if (prompt_file.len > 0) tiger.alloc_host_zeroed(u8, 32 * 1024 * 1024) else null;
    defer if (prompt_file_buf) |b| tiger.free_host_slice(u8, b);
    if (prompt_file_buf) |b| {
        prompt_text = read_file_into(prompt_file, b) orelse {
            tiger.print_stderr("Failed to read --prompt-file '{s}'\n", .{prompt_file});
            return 1;
        };
    }

    // Fast tokenizer-only diagnostic & benchmark command (no GPU required)
    if (std.mem.eql(u8, command, "tokenize")) {
        var tok: tokenizer_mod.Tokenizer = .{};
        if (!tok.load(model_directory)) return 1;
        defer tok.deinit();

        var scratch = tokenizer_mod.TokenizerScratch.init();
        defer scratch.deinit();

        const render_buf = tiger.alloc_host_zeroed(u8, 32 * 1024 * 1024);
        defer tiger.free_host_slice(u8, render_buf);
        const ids_buf = tiger.alloc_host_zeroed(i32, cli_tokens_max);
        defer tiger.free_host_slice(i32, ids_buf);

        var input_slice: []const u8 = prompt_text;
        if (!raw_prompt) {
            var rw = tiger.ByteWriter.init(render_buf);
            tokenizer_mod.Tokenizer.render_single_user_prompt(&rw, prompt_text, effort);
            input_slice = rw.slice();
        }
        const t0 = tiger.clock_monotonic_ns();
        const count = tok.encode_into(input_slice, true, &scratch, ids_buf);
        const enc_us: f64 = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - t0)) / 1000.0;
        tiger.print_stdout("[tokenize] count={d} encode_us={d:.1}\n", .{ count, enc_us });
        print_token_csv("IDS:", ids_buf[0..count]);
        return 0;
    }

    const need_tokenizer: bool = std.mem.eql(u8, command, "serve") or prompt_text.len > 0;
    var tokenizer: tokenizer_mod.Tokenizer = .{};
    if (need_tokenizer) {
        if (!tokenizer.load(model_directory)) return 1;
    }
    defer if (need_tokenizer) tokenizer.deinit();

    const quant_mode: constants.QuantMode = parse_mode(mode_str);
    const kv_format: constants.KVFormat = if (std.mem.eql(u8, kv_str, "bf16"))
        .bf16
    else if (std.mem.eql(u8, kv_str, "fp8"))
        .fp8
    else if (std.mem.eql(u8, kv_str, "q4"))
        .q4
    else if (quant_mode == .exact)
        .bf16
    else if (ctx > 458_752)
        .q4
    else
        .fp8;

    if (ctx > 524_288 and quant_mode != .exact) {
        embed_on_host = true;
    }

    const prompt_ids_buf = tiger.alloc_host_zeroed(i32, cli_tokens_max);
    defer tiger.free_host_slice(i32, prompt_ids_buf);
    var prompt_len: usize = 0;

    if (prompt_text.len > 0 and !has_tokens_arg) {
        var scratch = tokenizer_mod.TokenizerScratch.init();
        defer scratch.deinit();
        const render_buf = tiger.alloc_host_zeroed(u8, 32 * 1024 * 1024);
        defer tiger.free_host_slice(u8, render_buf);

        var input_slice: []const u8 = prompt_text;
        if (!raw_prompt) {
            var rw = tiger.ByteWriter.init(render_buf);
            tokenizer_mod.Tokenizer.render_single_user_prompt(&rw, prompt_text, effort);
            input_slice = rw.slice();
        }
        prompt_len = tokenizer.encode_into(input_slice, true, &scratch, prompt_ids_buf);
    } else {
        if (tokens_file.len > 0) {
            const file_buf = tiger.alloc_host_zeroed(u8, 32 * 1024 * 1024);
            defer tiger.free_host_slice(u8, file_buf);
            const loaded = read_file_into(tokens_file, file_buf) orelse {
                tiger.print_stderr("Failed to read --tokens-file '{s}'\n", .{tokens_file});
                return 1;
            };
            prompt_len = parse_csv_tokens(loaded, prompt_ids_buf);
        } else {
            prompt_len = parse_csv_tokens(tokens_csv, prompt_ids_buf);
        }
    }

    // Validate token ID bounds before GPU launch (prevents OOB embedding gather on malformed CSV!)
    for (prompt_ids_buf[0..prompt_len]) |tid| {
        if (tid < 0 or tid >= @as(i32, @intCast(constants.vocab_padded_count))) {
            tiger.print_stderr("Invalid token id {d} (valid range: 0..{d})\n", .{ tid, constants.vocab_padded_count - 1 });
            return 1;
        }
    }
    const is_prompt_cmd = std.mem.eql(u8, command, "logits") or std.mem.eql(u8, command, "kld") or std.mem.eql(u8, command, "generate") or std.mem.eql(u8, command, "bench");
    if (is_prompt_cmd and (prompt_len == 0 or prompt_len >= ctx)) {
        tiger.print_stderr("Prompt length {d} must be in 1..{d} (increase --ctx)\n", .{ prompt_len, if (ctx > 0) ctx - 1 else 0 });
        return 1;
    }

    if (ablation_path.len > 0 and !model_name_explicit) {
        model_name = "inkling-2-mini-abliterated";
    }

    var model_weights: weights_mod.ModelWeights = .{};
    const load_mtp: bool = (mtp > 0 or std.mem.eql(u8, command, "bench") or std.mem.eql(u8, command, "longctx") or std.mem.eql(u8, command, "serve"));
    if (!model_weights.load(model_directory, quant_mode, load_mtp, embed_on_host, ablation_path)) return 1;
    defer model_weights.deinit();

    var engine: engine_mod.Engine = .{};
    engine.init(&model_weights, ctx, use_graphs, kv_format, share_loop_kv);
    engine.spec_threshold = spec_threshold;
    engine.use_prompt_lookup = use_prompt_lookup;
    engine.sample_params = .{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .seed = if (seed != 0) seed else tiger.clock_monotonic_ns(),
    };
    defer engine.deinit();

    if (std.mem.eql(u8, command, "serve")) {
        return server_mod.run_openai_server(&engine, &tokenizer, .{
            .host = host,
            .port = port,
            .model_name = model_name,
            .api_key = api_key,
            .default_effort = effort,
            .default_mtp = mtp,
            .default_temperature = temperature,
            .default_top_p = top_p,
            .default_top_k = top_k,
            .context_tokens_max = ctx,
            .raw_thinking = raw_thinking,
            .keep_history_thinking = keep_history_thinking,
        });
    }

    if (std.mem.eql(u8, command, "logits")) {
        if (out_path.len == 0 or prompt_len < 2) {
            tiger.print_stderr("Usage: squid5090 logits <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --out <file.bin> [--mode exact] [--kv bf16]\n", .{});
            return 1;
        }
        const vocab: usize = constants.vocab_unpadded_count;
        const n_pos: usize = prompt_len;
        const chunk_buf = tiger.alloc_host_zeroed(f32, constants.mtp_block_size_max * vocab);
        defer tiger.free_host_slice(f32, chunk_buf);

        var pz: [1024]u8 = undefined;
        if (out_path.len >= pz.len) {
            tiger.print_stderr("Path too long: '{s}'\n", .{out_path});
            return 1;
        }
        @memcpy(pz[0..out_path.len], out_path);
        pz[out_path.len] = 0;
        const fd = tiger.open(@ptrCast(&pz), tiger.O_WRONLY | tiger.O_CREAT | tiger.O_TRUNC, @as(c_uint, 0o644));
        if (fd < 0) {
            tiger.print_stderr("Failed to open --out '{s}' for writing\n", .{out_path});
            return 1;
        }
        defer _ = tiger.close(fd);

        const WriteCtx = struct {
            fd: i32,
            bytes_written: usize = 0,
            write_err: bool = false,
            fn on_chunk(ctx_ptr: *anyopaque, _: usize, chunk_logits: []const f32, _: usize) bool {
                const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
                const raw: []const u8 = @as([*]const u8, @ptrCast(chunk_logits.ptr))[0 .. chunk_logits.len * @sizeOf(f32)];
                if (!tiger.write_fd_all_checked(self.fd, raw)) {
                    self.write_err = true;
                    return false;
                }
                self.bytes_written += raw.len;
                return true;
            }
        };
        var wctx: WriteCtx = .{ .fd = fd };
        if (!engine.eval_logits_streaming(prompt_ids_buf[0..n_pos], chunk_buf, WriteCtx.on_chunk, @ptrCast(&wctx)) or wctx.write_err) {
            tiger.print_stderr("I/O error writing logits to '{s}'\n", .{out_path});
            return 1;
        }
        tiger.print_stdout("[logits] wrote {d} positions x {d} vocab ({d:.1} MiB) to {s} (mode={s}, kv={s})\n", .{
            n_pos, vocab, @as(f64, @floatFromInt(wctx.bytes_written)) / (1024.0 * 1024.0), out_path, quant_mode.name(), kv_format.name(),
        });
        return 0;
    }

    if (std.mem.eql(u8, command, "kld")) {
        if (ref_path.len == 0 or prompt_len < 2) {
            tiger.print_stderr("Usage: squid5090 kld <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --ref <ref_logits.bin> [--mode nvfp4] [--kv q4]\n", .{});
            return 1;
        }
        const vocab: usize = constants.vocab_unpadded_count;
        const n_pos: usize = prompt_len;
        const chunk_buf = tiger.alloc_host_zeroed(f32, constants.mtp_block_size_max * vocab);
        defer tiger.free_host_slice(f32, chunk_buf);

        var pz: [1024]u8 = undefined;
        if (ref_path.len >= pz.len) {
            tiger.print_stderr("Path too long: '{s}'\n", .{ref_path});
            return 1;
        }
        @memcpy(pz[0..ref_path.len], ref_path);
        pz[ref_path.len] = 0;
        const fd = tiger.open(@ptrCast(&pz), tiger.O_RDONLY);
        if (fd < 0) {
            tiger.print_stderr("Failed to open --ref '{s}'\n", .{ref_path});
            return 1;
        }
        defer _ = tiger.close(fd);

        const ref_row = tiger.alloc_host_zeroed(f32, vocab);
        defer tiger.free_host_slice(f32, ref_row);
        const kld_vals = tiger.alloc_host_zeroed(f64, n_pos);
        defer tiger.free_host_slice(f64, kld_vals);

        const KldCtx = struct {
            fd: i32,
            vocab: usize,
            n_pos: usize,
            prompt_ids: []const i32,
            ref_row: []f32,
            kld_vals: []f64,
            sum_kld: f64 = 0.0,
            sum_nll_ref: f64 = 0.0,
            sum_nll_cand: f64 = 0.0,
            nll_cnt: usize = 0,
            top1_match: usize = 0,
            eof_err: bool = false,

            fn on_chunk(ctx_ptr: *anyopaque, pos_offset: usize, chunk_logits: []const f32, count: usize) bool {
                const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
                const row_bytes: usize = self.vocab * @sizeOf(f32);
                const ref_bytes_ptr: [*]u8 = @ptrCast(self.ref_row.ptr);
                var b: usize = 0;
                while (b < count) : (b += 1) {
                    const p_i = pos_offset + b;
                    var got: usize = 0;
                    while (got < row_bytes) {
                        const n = tiger.read(self.fd, ref_bytes_ptr + got, row_bytes - got);
                        if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
                        if (n <= 0) {
                            tiger.print_stderr("Unexpected EOF reading --ref at pos {d}\n", .{p_i});
                            self.eof_err = true;
                            return false;
                        }
                        got += @intCast(n);
                    }
                    const c_row = chunk_logits[b * self.vocab ..][0..self.vocab];

                    var max_r: f64 = -1e30;
                    var max_c: f64 = -1e30;
                    var arg_r: usize = 0;
                    var arg_c: usize = 0;
                    var v: usize = 0;
                    while (v < self.vocab) : (v += 1) {
                        const rv: f64 = self.ref_row[v];
                        const cv: f64 = c_row[v];
                        if (rv > max_r) {
                            max_r = rv;
                            arg_r = v;
                        }
                        if (cv > max_c) {
                            max_c = cv;
                            arg_c = v;
                        }
                    }
                    if (arg_r == arg_c) self.top1_match += 1;

                    var z_r: f64 = 0.0;
                    var z_c: f64 = 0.0;
                    v = 0;
                    while (v < self.vocab) : (v += 1) {
                        z_r += @exp(@as(f64, self.ref_row[v]) - max_r);
                        z_c += @exp(@as(f64, c_row[v]) - max_c);
                    }
                    const log_zr = max_r + @log(z_r);
                    const log_zc = max_c + @log(z_c);

                    var kl: f64 = 0.0;
                    v = 0;
                    while (v < self.vocab) : (v += 1) {
                        const log_p = @as(f64, self.ref_row[v]) - log_zr;
                        const log_q = @as(f64, c_row[v]) - log_zc;
                        const p_prob = @exp(log_p);
                        kl += p_prob * (log_p - log_q);
                    }
                    if (kl < 0.0) kl = 0.0;
                    self.kld_vals[p_i] = kl;
                    self.sum_kld += kl;

                    if (p_i + 1 < self.n_pos) {
                        const nxt: usize = @intCast(self.prompt_ids[p_i + 1]);
                        if (nxt < self.vocab) {
                            self.sum_nll_ref += -(@as(f64, self.ref_row[nxt]) - log_zr);
                            self.sum_nll_cand += -(@as(f64, c_row[nxt]) - log_zc);
                            self.nll_cnt += 1;
                        }
                    }
                }
                return true;
            }
        };

        var kctx: KldCtx = .{
            .fd = fd,
            .vocab = vocab,
            .n_pos = n_pos,
            .prompt_ids = prompt_ids_buf[0..n_pos],
            .ref_row = ref_row,
            .kld_vals = kld_vals,
        };
        if (!engine.eval_logits_streaming(prompt_ids_buf[0..n_pos], chunk_buf, KldCtx.on_chunk, @ptrCast(&kctx)) or kctx.eof_err) {
            return 1;
        }

        // Insertion sort kld_vals for median & p95
        var si: usize = 1;
        while (si < n_pos) : (si += 1) {
            const key = kld_vals[si];
            var sj: usize = si;
            while (sj > 0 and kld_vals[sj - 1] > key) : (sj -= 1) kld_vals[sj] = kld_vals[sj - 1];
            kld_vals[sj] = key;
        }
        const mean_kld = kctx.sum_kld / @as(f64, @floatFromInt(n_pos));
        const med_kld = kld_vals[n_pos / 2];
        const p95_kld = kld_vals[(n_pos * 95) / 100];
        const max_kld = kld_vals[n_pos - 1];
        const ppl_ref = @exp(kctx.sum_nll_ref / @as(f64, @floatFromInt(@max(kctx.nll_cnt, 1))));
        const ppl_cand = @exp(kctx.sum_nll_cand / @as(f64, @floatFromInt(@max(kctx.nll_cnt, 1))));
        const top1_pct = 100.0 * @as(f64, @floatFromInt(kctx.top1_match)) / @as(f64, @floatFromInt(n_pos));

        tiger.print_stdout(
            "[kld] mode={s} kv={s} pos={d} | mean_KLD={d:.6} med={d:.6} p95={d:.6} max={d:.6} | top1={d:.2}% | ref_PPL={d:.4} cand_PPL={d:.4} (dPPL={d:.4})\n",
            .{ quant_mode.name(), kv_format.name(), n_pos, mean_kld, med_kld, p95_kld, max_kld, top1_pct, ppl_ref, ppl_cand, ppl_cand - ppl_ref },
        );
        return 0;
    }

    if (std.mem.eql(u8, command, "longctx")) {
        const test_positions = [_]u32{ 1024, 16384, 32768, 65536, 131072, 262144, 393216, 524288, 786432, 1048000 };
        const total_vram_gib: f64 = @as(f64, @floatFromInt(model_weights.arena.capacity_bytes + engine.arena.capacity_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "=== LONG-CONTEXT SCALING (zig, mode={s}, kv={s}, max_ctx={d}, total_VRAM={d:.2} GiB) ===\n",
            .{ quant_mode.name(), kv_format.name(), ctx, total_vram_gib },
        );
        tiger.print_stdout("  {s:>9} | {s:>11} | {s:>11} | {s:>11} | {s:>13}\n", .{ "pos", "B=1 ms/tok", "B=1 tok/s", "B=4 ms/step", "B=4 @dl=2.5" });
        for (test_positions) |pos_val| {
            if (pos_val + 256 >= ctx) continue;
            const ms1 = engine.bench_step_at_pos(@intCast(pos_val), 0, 24);
            const ms4 = engine.bench_step_at_pos(@intCast(pos_val), 3, 24);
            tiger.print_stdout(
                "  {d:>9} | {d:>8.2} ms | {d:>7.1} t/s | {d:>8.2} ms | {d:>9.1} t/s\n",
                .{ pos_val, ms1, 1000.0 / ms1, ms4, 2500.0 / ms4 },
            );
        }
        return 0;
    }

    const out_tokens_buf = tiger.alloc_host_zeroed(i32, @max(max_tokens, 64));
    defer tiger.free_host_slice(i32, out_tokens_buf);

    if (std.mem.eql(u8, command, "generate")) {
        const st = engine.generate(prompt_ids_buf[0..prompt_len], max_tokens, mtp, out_tokens_buf, null, null);
        const dec_toks: f64 = @floatFromInt(if (st.generated_tokens > 0) st.generated_tokens - 1 else 0);
        const tps: f64 = if (st.decode_ms > 0.0) (dec_toks * 1000.0 / st.decode_ms) else 0.0;
        const acc_pct: f64 = if (st.total_drafts > 0) (100.0 * @as(f64, @floatFromInt(st.total_accepted)) / @as(f64, @floatFromInt(st.total_drafts))) else 0.0;
        const dl: f64 = if (st.spec_steps > 0) (dec_toks / @as(f64, @floatFromInt(st.spec_steps))) else 1.0;

        tiger.print_stdout(
            "[gen] prompt={d} gen={d} prefill={d:.2}ms decode={d:.2}ms speed={d:.2} tok/s | mtp={d} acc={d:.1}% dl={d:.2} lookup={d}/{d}\n",
            .{ st.prompt_tokens, st.generated_tokens, st.prefill_ms, st.decode_ms, tps, mtp, acc_pct, dl, st.lookup_steps, st.spec_steps },
        );
        print_token_csv("TOKENS:", out_tokens_buf[0..st.generated_tokens]);
        if (need_tokenizer) print_decoded(&tokenizer, out_tokens_buf[0..st.generated_tokens]);
        return 0;
    }

    if (std.mem.eql(u8, command, "bench")) {
        const clamped_runs: usize = @min(@max(runs, 1), bench_runs_max);
        // 1 warmup run (also captures CUDA Graph)
        _ = engine.generate(prompt_ids_buf[0..prompt_len], @min(32, max_tokens), mtp, out_tokens_buf, null, null);

        var speeds: [bench_runs_max]f64 = undefined;
        var last_acc: f64 = 0.0;
        var last_dl: f64 = 1.0;
        var last_prefill: f64 = 0.0;
        var sample_len: usize = 0;
        const sample_tokens = tiger.alloc_host_zeroed(i32, @max(max_tokens, 64));
        defer tiger.free_host_slice(i32, sample_tokens);

        var r: usize = 0;
        while (r < clamped_runs) : (r += 1) {
            if (temperature > 0.0 and seed == 0) {
                engine.sample_params.seed = tiger.hash_pair_u64(tiger.clock_monotonic_ns() ^ @as(u64, r));
            }
            const st = engine.generate(prompt_ids_buf[0..prompt_len], max_tokens, mtp, out_tokens_buf, null, null);
            const dec_toks: f64 = @floatFromInt(if (st.generated_tokens > 0) st.generated_tokens - 1 else 0);
            const tps: f64 = if (st.decode_ms > 0.0) (dec_toks * 1000.0 / st.decode_ms) else 0.0;
            const acc: f64 = if (st.total_drafts > 0) (100.0 * @as(f64, @floatFromInt(st.total_accepted)) / @as(f64, @floatFromInt(st.total_drafts))) else 0.0;
            const dl: f64 = if (st.spec_steps > 0) (dec_toks / @as(f64, @floatFromInt(st.spec_steps))) else 1.0;
            speeds[r] = tps;
            last_acc = acc;
            last_dl = dl;
            last_prefill = st.prefill_ms;
            if (r == 0) {
                sample_len = st.generated_tokens;
                @memcpy(sample_tokens[0..sample_len], out_tokens_buf[0..sample_len]);
            }
            tiger.print_stdout(
                "  run {d}/{d}: gen={d} decode={d:.2}ms speed={d:.2} tok/s (acc={d:.1}% dl={d:.2})\n",
                .{ r + 1, clamped_runs, st.generated_tokens, st.decode_ms, tps, acc, dl },
            );
        }

        // Insertion sort small runs array (<= 64 elements)
        var si: usize = 1;
        while (si < clamped_runs) : (si += 1) {
            const v = speeds[si];
            var sj: usize = si;
            while (sj > 0 and speeds[sj - 1] > v) : (sj -= 1) speeds[sj] = speeds[sj - 1];
            speeds[sj] = v;
        }
        const median: f64 = if (clamped_runs % 2 == 1)
            speeds[clamped_runs / 2]
        else
            0.5 * (speeds[clamped_runs / 2 - 1] + speeds[clamped_runs / 2]);

        tiger.print_stdout(
            "MEDIAN: {d:.2} tok/s | min={d:.2} max={d:.2} | prefill={d:.2}ms | acc={d:.1}% dl={d:.2} | mode={s} kv={s} temp={d:.2} mtp={d} graph={d}\n",
            .{ median, speeds[0], speeds[clamped_runs - 1], last_prefill, last_acc, last_dl, quant_mode.name(), kv_format.name(), temperature, mtp, @as(u8, if (use_graphs) 1 else 0) },
        );
        print_token_csv("TOKENS:", sample_tokens[0..sample_len]);
        if (need_tokenizer) print_decoded(&tokenizer, sample_tokens[0..sample_len]);
        return 0;
    }

    tiger.print_stderr("Unknown command '{s}'\n", .{command});
    return 1;
}

fn parse_mode(s: []const u8) constants.QuantMode {
    if (std.mem.eql(u8, s, "exact")) return .exact;
    if (std.mem.eql(u8, s, "fp8")) return .fp8;
    if (std.mem.eql(u8, s, "nvfp4")) return .nvfp4;
    tiger.print_stderr("Unknown mode '{s}' (expected exact|fp8|nvfp4)\n", .{s});
    @panic("Invalid quantization mode");
}

fn parse_csv_tokens(csv: []const u8, out: []i32) usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (pos < csv.len and count < out.len) {
        while (pos < csv.len and (csv[pos] == ',' or csv[pos] == ' ' or csv[pos] == '\n' or csv[pos] == '\r' or csv[pos] == '\t')) : (pos += 1) {}
        if (pos >= csv.len) break;
        const start = pos;
        while (pos < csv.len and csv[pos] != ',' and csv[pos] != ' ' and csv[pos] != '\n' and csv[pos] != '\r' and csv[pos] != '\t') : (pos += 1) {}
        if (std.fmt.parseInt(i32, csv[start..pos], 10)) |v| {
            out[count] = v;
            count += 1;
        } else |_| {}
    }
    return count;
}

fn read_file_into(path: []const u8, buf: []u8) ?[]const u8 {
    var pz: [1024]u8 = undefined;
    if (path.len >= pz.len) return null;
    @memcpy(pz[0..path.len], path);
    pz[path.len] = 0;
    const fd = tiger.open(@ptrCast(&pz), tiger.O_RDONLY);
    if (fd < 0) return null;
    defer _ = tiger.close(fd);
    var total: usize = 0;
    var iter: u32 = 0;
    while (total < buf.len and iter < 65536) : (iter += 1) {
        const n = tiger.read(fd, buf[total..].ptr, buf.len - total);
        if (n < 0 and tiger.get_errno() == tiger.EINTR) continue;
        if (n <= 0) break;
        total += @intCast(n);
    }
    return buf[0..total];
}

fn print_token_csv(prefix: []const u8, tokens: []const i32) void {
    tiger.write_fd_all(1, prefix);
    var chunk: [4096]u8 = undefined;
    var w = tiger.ByteWriter.init(&chunk);
    for (tokens, 0..) |tok, idx| {
        if (w.remaining() < 24) {
            tiger.write_fd_all(1, w.slice());
            w.reset();
        }
        w.append_fmt("{s}{d}", .{ if (idx == 0) " " else ",", tok });
    }
    w.append_byte('\n');
    tiger.write_fd_all(1, w.slice());
}

fn print_decoded(tok: *const tokenizer_mod.Tokenizer, tokens: []const i32) void {
    tiger.write_fd_all(1, "\n--- DECODED OUTPUT ---\n");
    for (tokens) |tid| {
        const bytes = tok.decode_token_bytes(tid, false);
        if (bytes.len > 0) tiger.write_fd_all(1, bytes);
    }
    tiger.write_fd_all(1, "\n----------------------\n");
}

fn print_usage() void {
    tiger.write_fd_all(2,
        "squid5090 — TigerStyle Zig + Blackwell CUDA Inference Engine & OpenAI Server for Inkling-2-Mini\n\n" ++
            "Usage:\n" ++
            "  squid5090 serve    <model_dir> [--host 0.0.0.0] [--port 8000] [--mode fp8|nvfp4|exact] [--kv auto|bf16|fp8|q4] [--share-loop-kv] [--ctx 65536] [--mtp 7] [--spec-threshold 0.80] [--temp 1.0] [--top-p 0.95] [--top-k 0] [--keep-history-thinking] [--ablation PATH] [--model-name inkling-2-mini] [--effort none] [--api-key KEY]\n" ++
            "  squid5090 generate <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens CSV | --tokens-file PATH) [--mode exact] [--kv auto] [--share-loop-kv] [--mtp 0] [--spec-threshold 0.80] [--temp 0.0] [--top-p 0.95] [--top-k 0] [--seed N] [--ablation PATH] [--max-tokens 128] [--ctx 8192]\n" ++
            "  squid5090 bench    <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens CSV) [--mode exact] [--kv auto] [--share-loop-kv] [--mtp 0] [--spec-threshold 0.80] [--temp 0.0] [--top-p 0.95] [--ablation PATH] [--max-tokens 128] [--runs 6] [--ctx 8192]\n" ++
            "  squid5090 longctx  <model_dir> [--mode exact|fp8|nvfp4] [--kv fp8|q4|bf16] [--share-loop-kv] [--ctx 1048576]\n" ++
            "  squid5090 logits   <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --out FILE [--mode exact] [--kv bf16] [--share-loop-kv] [--ctx 8192]\n" ++
            "  squid5090 kld      <model_dir> (--prompt TEXT | --prompt-file PATH | --tokens-file PATH) --ref FILE [--mode exact|fp8|nvfp4] [--kv auto|bf16|fp8|q4] [--share-loop-kv] [--ctx 8192]\n" ++
            "  squid5090 tokenize <model_dir> (--prompt TEXT | --prompt-file PATH) [--raw]\n",
    );
}

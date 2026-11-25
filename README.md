# squid5090

**A single-binary, zero-Python TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible HTTP/SSE server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an NVIDIA RTX 5090.**

> **Upfront Attribution:** Several core Blackwell kernel designs and features in `squid5090` — including Programmatic Dependent Launch (PDL) overlap, 16-CTA cluster DSMEM/`mbarrier` BF16 controllers & router, 1-launch cluster DSMEM decode attention (`attn_cluster_decode_kernel`), spill-free 16-column `U=3` FP8/NVFP4 `mma.m16n8k16` GEMVs & LM-head, 128-CTA multi-block stochastic LM-head sampling with 64-bit Gumbel, Top-K `1024` radix Markov drafting with a dynamic context shortlist, the `128×128×64` `cp.async` MoE prefill pipeline, device-side `{2, 4, 8}` CUDA Graph conditional switch drafting (`cudaGraphCondTypeSwitch`), and the 1.8 MB runtime-gated Fisher-LDA abliteration format & tooling — were adapted from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (Apache-2.0). Hybrid prompt-lookup drafting was inspired by [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine), and the roofline workflow follows [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill). Full attributions live in [`NOTICE`](NOTICE).

![squid5090 RTX 5090 Benchmarks, KLD Frontier & Long-Context Scaling](assets/benchmark.png)

> **Deep Dives:**
> - **[`LITE.md`](LITE.md)** — Plain-English, 3Blue1Brown-style animated guide to GPU roofline optimization, speculative decoding, and geometric Fisher-LDA abliteration.
> - **[`LEARNINGS.md`](LEARNINGS.md)** — Complete technical reference: CUDA kernels, roofline math, KLD/quality analysis, remaining optimization headroom, and the `uv` abliteration pipeline.

> **Running on a 16 GiB RTX 5080?** See **[`PORT-5080.md`](PORT-5080.md)** (CUDA 12.8+ builds, WSL2, VRAM auto-fit).

---

## Benchmarks & Quality (Single RTX 5090, `c=1`)

All numbers below are measured directly on one RTX 5090 (32 GiB GDDR7) with `squid5090`'s built-in `bench`, `kld`, and `generate` commands. Every precision mode is evaluated for **exact FP64 KL divergence vs. the unmodified `exact / bf16` checkpoint** on the checked-in [`tools/kld_corpus.txt`](tools/kld_corpus.txt) (**7,784 tokens**, 7.6× past the 1,024-token local sliding window so all 8 quantized global KV layers are directly tested).

### 1. Precision × Long-Context Quality (KLD) × Decode Throughput (`mtp=7`, Adaptive `{2,4,8}` + Prompt-Lookup)

| Mode / KV | Mean KLD *(7,784 tok)* | Top-1 Match | PPL *(Ref `2.7028`)* | Code (`T=0.0` / `T=1.0`) | Code-Copy (`T=0.0` / `T=1.0`) | Chat (`T=0.0` / `T=1.0`) | Prose (`T=0.0` / `T=1.0`) | Weight VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`exact / bf16`** *(official)* | **0.0000** | 100.00% | 2.7028 | 246.7 / 225.2 tok/s | 419.2 / 237.2 tok/s | 229.0 / 217.1 tok/s | 201.9 / 191.0 tok/s | 18.28 GiB |
| `exact / fp8` | 0.0035 | 98.39% | 2.7007 | — | — | — | — | 18.28 GiB |
| `exact / q4` | 0.0078 | 97.15% | 2.7063 | — | — | — | — | 18.28 GiB |
| **`fp8 / fp8`** *(default ≤448K)* | **0.0058** | **97.53%** | 2.6954 | **349.4 / 356.8 tok/s** | **438.3 / 456.4 tok/s** | 264.9 / **313.5 tok/s** | 270.1 / 272.1 tok/s | 17.08 GiB |
| **`fp8 / q4`** *(default 512K)* | **0.0094** | **96.79%** | 2.7049 | **394.2 / 293.9 tok/s** | **455.4 / 332.4 tok/s** | **298.2 / 278.6 tok/s** | **288.0 / 240.7 tok/s** | 17.08 GiB |
| `nvfp4 / fp8` | 0.0230 | 94.99% | 2.7485 | — | — | — | — | 14.66 GiB |
| **`nvfp4 / q4`** *(opt-in max speed / 1M)* | **0.0268** | **94.75%** | 2.7569 | **383.0 / 366.2 tok/s** | **479.7 / 573.5 tok/s** | **360.9 / 342.4 tok/s** | **303.6 / 298.6 tok/s** | 14.66 GiB |

- **Accuracy-first default (`--mode fp8`):** `squid5090 serve` defaults to **`--mode fp8`** (keeping ALL 24 MoE expert layers in 100% official checkpoint precision — L0–19 NVFP4 + L20–23 MXFP8 — plus official BF16 routers/controllers in 16-CTA DSMEM cluster kernels, 1-launch cluster decode attention, spill-free 16-col `U=3` FP8 MMA attention + `lm_head`, native `sm_120a` block-scaled MoE + FP8 `cublasLt` prefill, and FP8 DSpark with Top-K `1024` exact BF16 Markov folding + dynamic context shortlist). This delivers **`0.0058–0.0094` long-context KLD** (`ΔPPL -0.0073` to `+0.0021` over 7,784 tokens — **26× smaller PPL delta** than 4-bit) while reaching **`456.4 tok/s`** `T=1.0` code-copy, **`356.8–394.2 tok/s`** code, **`313.5 tok/s`** `T=1.0` chat, and **`16,855 tok/s` HTTP / `18,318 tok/s` CLI** cold 18.9K prefill (`20,007 tok/s` `B=2048` chunk peak)!
- **Temperature transparency:** `bench` defaults to `--temp 0.0` (greedy); `serve` defaults to `--temp 1.0 --top-p 0.95` (matching Inkling-2-Mini's official `generation_config.json`), using **exact Leviathan/Chen stochastic rejection sampling** (`q_i(v)` drafted from DSpark + residual resampling from $\mathrm{norm}(\max(0, p - q))$ inside the CUDA Graph). `PromptLookupIndex` automatically stays active for greedy `T=0` copy bursts and defers to DSpark at `T > 0` so stochastic rejection sampling never hits a lookup mismatch cliff.
- **Clock-state & prompt transparency:** The unified table above was measured on identical prompts at shared-GPU `P1` memory clocks (`13,801 MHz`); dedicated `P0` clocks (`14,001 MHz`) yield ~7% higher decode on the same prompts (`589.5 tok/s` `nvfp4` code, `332.5 tok/s` chat, `541.1 tok/s` ablated, `259.6 tok/s` `mtp=0`). On longer verbatim RAG/code-copy bursts (`91.5%` prompt-lookup hit rate, `P0`), `nvfp4` hybrid lookup reaches `771.6 tok/s`.
- **Short- vs. long-context KLD:** On 529-token prompts (inside the 1,024-token BF16 local window), KLD is `0.0030` (`fp8/fp8`), `0.0050` (`fp8/q4`), and `0.0184` (`nvfp4/q4`). At 7,784 tokens (7.6× past the local window, shown above), `q4` global KV adds only `+0.0035` KLD over `fp8` KV (`ΔPPL +0.0035` in `fp8/q4`).
- **Choosing a mode:** Keep the default **`--mode fp8`** (`--kv auto` selects `fp8` ≤ 448K and `q4` at 512K–1M) for near-lossless accuracy (`0.006–0.0096` KLD, `~422–487 tok/s`); opt in to **`--mode nvfp4 --kv q4`** only when you need peak 4-bit throughput (`0.0263` KLD, `ΔPPL +0.0559`) or 1M context in `23.2 GiB`.

### 2. Real Long-Context Recall vs. Synthetic 1M Step-Latency Scaling (`--mode nvfp4 --kv q4`)

Real prefilled single-needle passkey retrieval (planted at 50% prompt depth) is reported separately from synthetic zeroed-KV step latency (`squid5090 longctx`):

| Context Depth | Test Type | Prefill Latency / Rate | Decode Speed (`c=1`) | Total VRAM (0 spill) | Result / Note |
|---|---|---:|---:|---:|---|
| **20K** (`19,978` tok) | Real cold prefill | `2,282 ms` (`8,755 tok/s`) | 453.6 tok/s | 16.7 GiB | `9,181 tok/s` `B=2048` chunk peak |
| **64K** (`65,530` tok) | **Real prefilled Needle (50% depth)** | `8.46 s` (`7,742 tok/s`) | 318–331 tok/s | 16.7 GiB | **Verbatim recall PASS** |
| **128K** (`131,057` tok) | **Real prefilled Needle (50% depth)** | `21.89 s` (`5,988 tok/s`) | 166.6 tok/s | 17.4 GiB | **Verbatim recall PASS** (`COBALT-FALCON-7492`) |
| **256K** (`262,115` tok) | **Real prefilled Needle (50% depth)** | `57.29 s` (`4,575 tok/s`) | 123.9 tok/s | 17.9 GiB | **Verbatim recall PASS** (`OBSIDIAN-HERON-3815`) |
| **512K** (`523,961` tok) | **Real prefilled Needle (50% depth)** | `171.13 s` (`3,062 tok/s`) | 74.4 tok/s | 20.3 GiB | **Verbatim recall PASS** (`VERMILION-LYNX-9064`) |
| **1M** (`1,048,000` tok) | *Synthetic `longctx` (zeroed KV)* | — | **45.7 tok/s** (`B=1`, `21.86 ms/tok`)<br>*102.4 P1 / 108.0 P0 (`B=4 @dl=2.5`)* | **23.23 GiB** | Real KV allocation & step latency; `@dl=2.5` is synthetic multiplier |
| **Warm 16K / 82.5K turn** | Real HTTP prefix cache | **`71.9 ms` / `62.4 ms`** | 446.7 / 237.5 tok/s | 23.2 GiB | Host-RAM local checkpoints (`0` extra VRAM); `32 MiB` static HTTP buffer |

### 3. LoopCD-Hidden Contrastive Decoding ([`arXiv:2610.02185`](https://arxiv.org/abs/2610.02185))

Inkling-2-Mini runs its 12-layer recurrent core (`L6..L17`) for $R = 3$ passes ($h_1 \to h_2 \to h_3$) before entering the 6-layer Coda (`L18..L23`). `squid5090` implements **LoopCD-Hidden** (Eq. 2 of Liu et al., 2026) directly inside the existing Loop→Coda mega-fusion kernel (`mega_mlp_post_next_depth_norm_kernel` at visit 42) with **zero extra CUDA kernel launches**, **zero extra KV cache**, **zero CUDA Graph rebuilds** on per-request changes, and automatic prefix-cache invalidation whenever `loopcd_omega` or `loopcd_ref` changes:

$$h' = h_3 + \omega\,(h_3 - h_b), \qquad b \in \{1, 2\}$$

- **Zero-overhead & lossless verification:** When $\omega = 0.0$ (default), the kernel skips the `fmaf` cleanly (`0.000000` delta vs. baseline; `fp8/q4` KLD stays `0.009574`). When $\omega > 0$, `exact` mode retains 100% bit-exact `mtp=0 == mtp=7` parity, and `fp8/q4` (`T=1.0`, `mtp=7`) decode throughput stays equal or slightly faster (`247.5 → 260.4 tok/s` at $\omega = 0.2$, `acc: 49.2% → 52.5%`).
- **Empirical RTX 5090 sweep (`fp8/q4`, 30-task Math + Code + Logic benchmark & 7,784-tok KLD):**
  - **Thinking mode (`enable_thinking=true`, default):** $\omega \in \{0.1, 0.2\}$ with $h_b = h_1$ (and $\omega = 0.3$ with $h_2$) lifts accuracy from **`25/30 (83.3%)` → `27/30 (90.0%, +6.7 pts)`** (`10/10` Math, `10/10` Logic, `7/10` Code), with $\omega = 0.1$ using **5.5% fewer thinking tokens** (`11,893` vs `12,584`) and raising 7,784-tok top-1 match (`96.93% → 96.96%`, `ΔPPL +0.0065`). Above $\omega \ge 0.5$, generation degrades (`24/30`), matching Appendix G.1 of the paper.
  - **Direct mode (`enable_thinking=false`):** Unmodified baseline ($\omega = 0.0$) scores `29/30 (96.7%)` vs `27–28/30` at $\omega \in [0.2, 0.5]$, so `squid5090` keeps `--loopcd-omega 0.0` as the opt-in default and exposes `--loopcd-omega` / `--loopcd-ref` on both CLI and per-request HTTP JSON (`"loopcd_omega": 0.1, "loopcd_ref": 1`).

### 4. Live Same-Hardware Cross-Engine Benchmark & Accuracy (1× RTX 5090)

![Live Cross-Engine Benchmark & 30-Task Accuracy on 1× RTX 5090](assets/engine_comparison.png)

> **100% live apples-to-apples methodology:** Every single row below — **`ingine`** (built with isolated CUDA 13.4), **`basalt v0.5.7`** (`MX8` pack converted directly from the same checkpoint), and all 4 **`squid5090`** modes — was measured **live back-to-back on the exact same RTX 5090** (`driver 590.48.01`, `P1` clocks `13,801 MHz`, identical unmodified Inkling-2-Mini weights, identical 4 HTTP SSE prompts generating 512 tokens each at `T=0.0` and `T=1.0, top_p=0.95` median of 3 runs, identical cold 18,937-token HTTP SSE prefill, and identical 30-task Math + unit-tested Python Code + Logic accuracy suite at `reasoning_effort = low` and `none`). `squid5090` also passes bit-exact short & >1K long-prompt `exact/bf16` parity (`mtp=0 == mtp=7`).

| Engine / Mode (unmodified weights) | Chat `T=0 / T=1` | Prose `T=0 / T=1` | Code `T=0 / T=1` | Copy `T=0 / T=1` | Cold 18.9K HTTP Prefill | 30-Task Acc (`low` / `none`) | VRAM Footprint* | Max Context (1× 5090) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`ingine`** (`fp8/fp8`, CUDA 13.4) | **361.1 / 368.7** | **346.9 / 302.3** | **442.3 / 525.8** | **632.5** / 542.5 | 10,985 tok/s | **29/30** (96.7%) / 27/30 (90.0%) | 20.8 GiB | 262K (`0` spill) |
| **`basalt v0.5.7`** (`MX8/int8`) | 320.3 / 276.0 | 284.4 / 264.6 | 365.5 / 303.0 | 452.9 / 378.8 | **17,494 tok/s** | **29/30** (96.7%) / 28/30 (93.3%) | 19.8 GiB | 1M *(RAM spill >256K)* |
| **`basalt v0.5.7`** (`MX8/fp8`) | 334.1 / 271.4 | 265.8 / 257.1 | 319.1 / 292.0 | 425.1 / 352.6 | 16,307 tok/s | **29/30** (96.7%) / 27/30 (90.0%) | 19.7 GiB | 1M *(RAM spill >256K)* |
| **`squid5090`** (`exact/bf16`) | 229.0 / 217.1 | 201.9 / 191.0 | 246.7 / 225.2 | 419.2 / 237.2 | 9,875 tok/s | **29/30** (96.7%) / **29/30 (96.7%)** | 20.7 GiB | 64K–128K (`0` spill) |
| **`squid5090`** (`fp8/fp8`, ≤448K def.) | 264.9 / 313.5 | 270.1 / 272.1 | 349.4 / 356.8 | 438.3 / 456.4 | **16,855 tok/s** *(CLI: 18,318)* | 28/30 (93.3%) / 27/30 (90.0%) | 18.9 GiB | 256K–448K (`0` spill) |
| **`squid5090`** (`fp8/q4`, 512K def.) | 298.2 / 278.6 | 288.0 / 240.7 | 394.2 / 293.9 | 455.4 / 332.4 | 15,632 tok/s *(CLI: 18,168)* | 28/30 (93.3%) / 28/30 (93.3%) | 18.6 GiB | **512K** (`22.8 GiB`, `0` spill) |
| **`squid5090`** (`nvfp4/q4`, opt-in 1M) | 360.9 / 342.4 | 303.6 / 298.6 | 383.0 / 366.2 | 479.7 / **573.5** | 15,048 tok/s | 28/30 (93.3%) / 28/30 (93.3%) | **16.2 GiB** | **1,048,576** (`23.5 GiB`, `0` spill) |

*\*VRAM = `squid5090` weight + runtime arenas at 32K ctx (`+0.25–0.50 GiB` at 64K); `ingine` and `basalt` measured net @64K. Pure `B=1` (`mtp=0`) `squid5090 fp8/fp8` runs at `247 tok/s` HTTP / `271.5 tok/s` CLI. For reference, `llama.cpp` (`IQ4_NL` GGUF, no DSpark draft, published in Basalt docs) runs at `92.5–94.5 tok/s` decode and `3,093 tok/s` prefill.*

- **Honest per-engine summary from our live RTX 5090 runs:**
  - **`ingine`** leads `fp8/fp8` greedy/code single-stream decode (`361–369` chat, `302–347` prose, `442–526` code, `633` greedy copy) and supports vision — at the cost of requiring CUDA 13.4, slower cold prefill (`10,985 tok/s`), higher VRAM (`20.8 GiB`), and a `262K` context ceiling.
  - **`basalt v0.5.7`** leads HTTP cold prompt prefill (`17,494 tok/s` `MX8/int8`, `16,307 tok/s` `MX8/fp8`) and leads feature breadth (**vision, constrained JSON grammar, parallel slots, multi-GPU, elastic RAM-spilled 1M context**).
  - **`squid5090`** now **beats `basalt v0.5.7` on all 8 of 8 decode benchmarks** in `nvfp4/q4` (`360.9/342.4` chat, `303.6/298.6` prose, `383.0/366.2` code, `479.7/573.5` copy — tying `ingine` on greedy chat and **beating `ingine` on `T=1.0` copy `573.5` vs `542.5`!**) and **beats `basalt v0.5.7` on all 4 `T=1.0` decode benchmarks in `fp8/fp8`** (`313.5` chat, `272.1` prose, `356.8` code, `456.4` copy), **matches `basalt` on cold 18.9K prefill** (`16,855 tok/s` HTTP `fp8/fp8` — **1.53× faster than `ingine`** and ahead of `basalt MX8/fp8`; **`18,168–18,318 tok/s` pure CLI**, `20,007 tok/s` `B=2048` chunk peak), leads **exact accuracy** (`exact/bf16` **`29/30` on both `low` and `none`**; `fp8/fp8` `0.0058` / `fp8/q4` `0.0094` 7,784-tok KLD), uses the **lowest VRAM** (`16.2–18.9 GiB`, enabling **zero-spill 512K `fp8/q4` and 1M `nvfp4/q4`**), ships as a **16 MiB zero-Python single binary** on CUDA 13.0/13.1 (`driver 580+`), and adds zero-overhead **LoopCD-Hidden**.

### Current Scope & Limitations
- **Text-only:** No vision/image encoder yet.
- **No constrained grammar / JSON-schema decoding** yet (standard OpenAI `tools` / `tool_calls` streaming is supported).
- **Single-user (`c=1`) slot architecture:** Optimized for lowest single-user latency and fitting up to 1M context in 23.2 GiB with `2 cudaMalloc` arenas and zero post-init host `malloc`; not a multi-user batching server.
- **HTTP request buffer:** Static `32 MiB` request & prompt-render buffers (handles 512K–1M-token JSON payloads; returns HTTP `413` above `32 MiB`).

---

## Quickstart (Linux Build & Install)

### 1. Linux Requirements

- **OS:** Linux x86_64 (tested on Ubuntu 24.04 / Debian 12–13)
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g., RTX 5090) with ≥ 16 GiB free VRAM (≥ 23.5 GiB for full 1M context)
- **NVIDIA Driver:** `580+` (tested on `590.48.01`)
- **CUDA Toolkit:** CUDA `13.0+` with `nvcc` and `cuBLAS` (tested on CUDA `13.1` at `/usr/local/cuda`)
- **Build Tools:** `g++`, `git`, `curl`, and **Zig `0.17.1+`**

```bash
sudo apt-get update && sudo apt-get install -y build-essential git curl
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version && zig version
```

*Runtime links only against `libc`, `libstdc++`, `libcudart`, `libcublas`, `libcublasLt`, and `libpthread` — zero Python, CMake, Make, or external packages.*

### 2. Clone, Build & Download Weights

```bash
git clone https://github.com/tensorfish/squid5090.git
cd squid5090
zig build                     # defaults to -Doptimize=fast (ReleaseFast); binary at zig-out/bin/squid5090

uvx --from huggingface-hub hf download thinkingmachines/Inkling-2-Mini \
  --local-dir /opt/models/inkling-2-mini
```

*(Optional draft speedup: if `draft_vocab_32k.bin` — a flat little-endian `int32` array of the top 32,768 frequent token IDs — is placed in the model directory, `squid5090` automatically prunes the DSpark draft LM head from 200K to 32K for ~2× faster drafting (`868 → 428 µs`). Without it, or with `SQUID_DRAFT_VOCAB=full`, `squid5090` seamlessly uses the full 200K draft vocabulary.)*

### 3. Run the OpenAI API Server, CLI, or KLD Evaluator

```bash
# Default high-accuracy OpenAI HTTP/SSE server (FP8 Tensor Core MMA + official L0-23 MoE, 512K Q4 KV, ~22.6 GiB VRAM, 0.0096 KLD)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode fp8 --kv q4 --ctx 524288 --mtp 7 --temp 1.0 --top-p 0.95

# Near-lossless 256K FP8 KV (0.0061 long-ctx KLD, dPPL -0.0030, ~22.6 GiB VRAM)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode fp8 --kv fp8 --ctx 262144 --mtp 7 --temp 1.0 --top-p 0.95

# Opt-in max-speed / 1M context NVFP4 mode (~23.2 GiB at 1M, 0.0263 KLD)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 --temp 1.0 --top-p 0.95

# Optional: enable 1.8 MB runtime-gated Fisher-LDA abliteration (zero weight modification)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode fp8 --kv q4 --ctx 524288 --mtp 7 \
  --ablation /opt/models/inkling-2-mini/inkling-2-mini-abliterated.safetensors

# Benchmark (supports --temp 0.0 or --temp 1.0)
./zig-out/bin/squid5090 bench /opt/models/inkling-2-mini \
  --prompt "Write a Python LRU cache with tests." --mode nvfp4 --kv q4 --mtp 7 --temp 1.0 --max-tokens 1024 --runs 3

# Reproduce exact FP64 long-context KLD (7,784 tokens) vs official exact/bf16
./zig-out/bin/squid5090 logits /opt/models/inkling-2-mini --prompt-file tools/kld_corpus.txt --raw --ctx 8192 --mode exact --kv bf16 --out /tmp/ref.bin
./zig-out/bin/squid5090 kld    /opt/models/inkling-2-mini --prompt-file tools/kld_corpus.txt --raw --ctx 8192 --ref /tmp/ref.bin --mode fp8   --kv q4
./zig-out/bin/squid5090 kld    /opt/models/inkling-2-mini --prompt-file tools/kld_corpus.txt --raw --ctx 8192 --ref /tmp/ref.bin --mode nvfp4 --kv q4
```

**Install as a persistent `systemd` service:**
```bash
sudo cp deploy/squid5090.service /etc/systemd/system/squid5090.service
sudo systemctl daemon-reload && sudo systemctl enable --now squid5090.service
curl -s http://localhost:8000/health
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `fp8` (`serve`), `exact` (CLI) | Weight precision: `exact` (BF16 dense) \| `fp8` (FP8 TC MMA dense + official L0–23 MoE, `0.006–0.0096` KLD) \| `nvfp4` (`0.026` KLD, max speed) |
| `--kv` | `auto` | Global KV cache format: `bf16` \| `fp8` \| `q4` (enables up to `1M` context in `23.2 GiB`) |
| `--ctx` | `65536` (`serve`), `8192` (CLI) | Context window size up to `1048576` |
| `--mtp` | `7` (`serve`), `0` (CLI) | DSpark drafter (`0` = off, `7` = adaptive `{2,4,8}` CUDA Graph switch) |
| `--spec-threshold` | `0.80` | Markov confidence threshold for `{2,4,8}` verify width (`0.0` = fixed `B=8`) |
| `--temp` / `--top-p` / `--top-k` | `1.0` / `0.95` / `0` (`serve`), `0.0` (CLI) | GPU sampling & exact Leviathan/Chen rejection sampling defaults (overridable per-request) |
| `--loopcd-omega` / `--loopcd-ref` | `0.0` / `1` | Opt-in LoopCD-Hidden ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)) strength $\omega \in [-1, 2]$ (try `0.1`–`0.2` for thinking) & ref pass `1` ($h_1$) or `2` ($h_2$); overridable per HTTP request |
| `--max-tokens` | `128` (CLI), `16384` (HTTP fallback) | Max generated tokens per request |
| `--keep-history-thinking` | off | Re-inject prior turns' `reasoning_content` (off by default to prevent multi-turn thinking copy loops) |
| `--no-prompt-lookup` | off | Disable hybrid O(1) prompt-lookup drafter |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |
| `--api-key` | `$SQUID_API_KEY` | Optional `Authorization: Bearer <key>` auth for `serve` (constant-time compare, arbitrary key length) |

---

## Documentation & Attribution

- **[`LITE.md`](LITE.md)** — High-level, 3Blue1Brown-style animated guide to LLM optimization and abliteration.
- **[`LEARNINGS.md`](LEARNINGS.md)** — Consolidated technical reference: GPU kernels, KLD analysis, roofline audit, unexploited levers, and the `uv` abliteration pipeline.
- **[`tools/kld_corpus.txt`](tools/kld_corpus.txt)** — Checked-in 7,784-token multi-domain KLD evaluation corpus.
- **[`ledger.jsonl`](ledger.jsonl)** — Machine-readable record of every benchmark, KLD sweep, and needle run.
- **[`LICENSE`](LICENSE)** / **[`NOTICE`](NOTICE)** — Apache-2.0 license and complete open-source attributions.

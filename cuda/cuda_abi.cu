// Thin extern "C" ABI bridge exposing Blackwell sm_120a CUDA kernels & runtime to TigerStyle Zig.
// Zero C++ heap allocations or host orchestration — all memory management, safetensors loading,
// tokenization, graph orchestration, and HTTP/SSE serving live in pure TigerStyle Zig.

#include "config.hpp"
#include "kernels.hpp"
#include "kernels_common.cuh"
#include "weights.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>

using namespace squid;

namespace {

// --- Weight Packing & Graph Helper Kernels ---

__global__ void z_bf16_amax_kernel(const __nv_bfloat16* __restrict__ x, size_t n, float* __restrict__ out_amax) {
    float local_max = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(x[i])));
    }
    local_max = warp_max(local_max);
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) atomicMax(reinterpret_cast<int*>(out_amax), __float_as_int(v));
    }
}

__device__ __forceinline__ uint8_t z_quantize_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    uint8_t code;
    if      (a < 0.25f) code = 0;
    else if (a < 0.75f) code = 1;
    else if (a < 1.25f) code = 2;
    else if (a < 1.75f) code = 3;
    else if (a < 2.50f) code = 4;
    else if (a < 3.50f) code = 5;
    else if (a < 5.00f) code = 6;
    else                code = 7;
    return sign | code;
}

__global__ void z_quant_bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ dst_w, uint8_t* __restrict__ dst_s,
    int N, int K, float global_scale) {
    size_t total_blocks = (size_t)N * (K / 16);
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;
    const __nv_bfloat16* row_in = src + bidx * 16;
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __bfloat162float(row_in[i]);
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    float raw_scale = (global_scale > 0.0f) ? fminf(bmax / (6.0f * global_scale), 448.0f) : 0.0f;
    __nv_fp8_e4m3 s_fp8(raw_scale);
    dst_s[bidx] = s_fp8.__x;
    float s_eff = static_cast<float>(s_fp8) * global_scale;
    float inv_s = (s_eff > 1e-30f) ? (1.0f / s_eff) : 0.0f;
    uint8_t* out_w = dst_w + bidx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_w[i] = z_quantize_e2m1(vals[2 * i] * inv_s) | (z_quantize_e2m1(vals[2 * i + 1] * inv_s) << 4);
    }
}

__global__ void z_quant_bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ dst_w, float* __restrict__ dst_row_scales,
    int N, int K) {
    int r = blockIdx.x;
    if (r >= N) return;
    const __nv_bfloat16* row_in = src + (size_t)r * K;
    float lmax = 0.0f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) lmax = fmaxf(lmax, fabsf(__bfloat162float(row_in[c])));
    lmax = warp_max(lmax);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = lmax;
    __syncthreads();
    __shared__ float inv_scale;
    if (warp == 0) {
        float v = warp_max((lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f);
        if (lane == 0) {
            float sc = fmaxf(v / 448.0f, 1e-12f);
            dst_row_scales[r] = sc;
            inv_scale = 1.0f / sc;
        }
    }
    __syncthreads();
    float isc = inv_scale;
    uint8_t* row_out = dst_w + (size_t)r * K;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        __nv_fp8_e4m3 q(__bfloat162float(row_in[c]) * isc);
        row_out[c] = q.__x;
    }
}

__global__ void z_mxfp8_expert_amax_kernel(
    const uint8_t* __restrict__ src_w, const uint8_t* __restrict__ src_s,
    float* __restrict__ exp_amax, int rows, int cols) {
    int e = blockIdx.x;
    size_t expert_elems = (size_t)rows * cols;
    const uint8_t* ew = src_w + (size_t)e * expert_elems;
    const uint8_t* es = src_s + (size_t)e * (expert_elems / 32);
    float lmax = 0.0f;
    for (size_t idx = threadIdx.x; idx < expert_elems; idx += blockDim.x) {
        float v = fabsf(cvt_e4m3_to_float(ew[idx]) * cvt_ue8m0_to_float(es[idx >> 5]));
        lmax = fmaxf(lmax, v);
    }
    lmax = warp_max(lmax);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float v = warp_max((lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f);
        if (lane == 0) exp_amax[e] = fmaxf(v / (448.0f * 6.0f), 1e-12f);
    }
}

__global__ void z_mxfp8_to_nvfp4_expert_kernel(
    const uint8_t* __restrict__ src_w, const uint8_t* __restrict__ src_s,
    const float* __restrict__ exp_global_scale, uint8_t* __restrict__ dst_w, uint8_t* __restrict__ dst_s,
    int n_experts, int rows, int cols) {
    size_t blocks_per_expert = (size_t)rows * (cols / 16);
    size_t total_blocks = (size_t)n_experts * blocks_per_expert;
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;
    int e = (int)(bidx / blocks_per_expert);
    float gscale = exp_global_scale[e];
    size_t elem_off = bidx * 16;
    float mx_scale = cvt_ue8m0_to_float(src_s[elem_off >> 5]);
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = cvt_e4m3_to_float(src_w[elem_off + i]) * mx_scale;
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    __nv_fp8_e4m3 s_fp8(fminf(bmax / (6.0f * gscale), 448.0f));
    dst_s[bidx] = s_fp8.__x;
    float s_eff = static_cast<float>(s_fp8) * gscale;
    float inv_s = (s_eff > 1e-30f) ? (1.0f / s_eff) : 0.0f;
    uint8_t* out_w = dst_w + bidx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_w[i] = z_quantize_e2m1(vals[2 * i] * inv_s) | (z_quantize_e2m1(vals[2 * i + 1] * inv_s) << 4);
    }
}

__global__ void z_compute_dspark_fold_bias_kernel(
    const __nv_bfloat16* __restrict__ unembed, const __nv_bfloat16* __restrict__ hidden_out_b,
    float* __restrict__ fold_b, int V, int K) {
    int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= V) return;
    int lane = threadIdx.x;
    const __nv_bfloat16* row = unembed + (size_t)v * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc = fmaf(__bfloat162float(row[k]), __bfloat162float(hidden_out_b[k]), acc);
    acc = warp_sum(acc);
    if (lane == 0) fold_b[v] = acc;
}

__global__ void z_compute_dspark_fold_w_kernel(
    const __nv_bfloat16* __restrict__ unembed, const __nv_bfloat16* __restrict__ hidden_out_w,
    __nv_bfloat16* __restrict__ fold_out, int V, int K, int R) {
    int v = blockIdx.x;
    if (v >= V) return;
    int r = threadIdx.x;
    __shared__ float u_row[HIDDEN];
    const __nv_bfloat16* u_ptr = unembed + (size_t)v * K;
    for (int k = r; k < K; k += 256) u_row[k] = __bfloat162float(u_ptr[k]);
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc = fmaf(u_row[k], __bfloat162float(hidden_out_w[(size_t)k * R + r]), acc);
    fold_out[(size_t)v * R + r] = __float2bfloat16(acc);
}

__global__ void z_split_dspark_rnn_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ w_sp, __nv_bfloat16* __restrict__ w_h, __nv_bfloat16* __restrict__ w_f) {
    int o = blockIdx.x;
    constexpr int K_IN = 2 * MTP_MARKOV_RANK + 2 * HIDDEN;
    const __nv_bfloat16* row = src + (size_t)o * K_IN;
    for (int c = threadIdx.x; c < K_IN; c += 256) {
        __nv_bfloat16 v = row[c];
        if (c < 512) w_sp[(size_t)o * 512 + c] = v;
        else if (c < 512 + HIDDEN) w_h[(size_t)o * HIDDEN + (c - 512)] = v;
        else w_f[(size_t)o * HIDDEN + (c - 512 - HIDDEN)] = v;
    }
}

__global__ void z_inc_pos_and_copy_tok_kernel(int32_t* d_pos, int32_t* d_tok_in, const int32_t* d_tok_out) {
    if (threadIdx.x == 0) { *d_pos += 1; *d_tok_in = *d_tok_out; }
}

__global__ void z_inc_pos_by_kernel(int32_t* d_pos, int delta) {
    if (threadIdx.x == 0) *d_pos += delta;
}

__global__ void z_assemble_verify_tokens_kernel(
    const int32_t* d_anchor, const int32_t* d_drafts, int32_t* d_tokens, int num_drafts) {
    int i = threadIdx.x;
    if (i == 0) d_tokens[0] = *d_anchor;
    else if (i <= num_drafts) d_tokens[i] = d_drafts[i - 1];
}

__global__ void z_adaptive_select_width_kernel(
    const int32_t* __restrict__ d_anchor,
    const int32_t* __restrict__ d_drafts,
    const float* __restrict__ d_draft_conf,
    float threshold,
    int32_t* __restrict__ d_tokens,
    int32_t* __restrict__ d_chosen_width,
    cudaGraphConditionalHandle cond_handle) {
    if (threadIdx.x != 0) return;
    int n_conf = 0;
    while (n_conf < MTP_MAX_DRAFTS && d_draft_conf[n_conf] >= threshold) {
        ++n_conf;
    }
    unsigned int case_idx = (n_conf <= 1) ? 0u : ((n_conf <= 3) ? 1u : 2u);
    int width = (case_idx == 0u) ? 2 : ((case_idx == 1u) ? 4 : 8);
    d_tokens[0] = *d_anchor;
    for (int i = 0; i < width - 1; ++i) d_tokens[1 + i] = d_drafts[i];
    *d_chosen_width = width;
    cudaGraphSetConditional(cond_handle, case_idx);
}

struct CAdaptiveGraph {
    cudaGraph_t root_graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphConditionalHandle handle{};
    cudaGraph_t bodies[3] = {nullptr, nullptr, nullptr};
};

} // namespace

// Flat C-ABI structs mirrored bit-for-bit in zig/src/cuda.zig!
extern "C" {

struct CLinearWeight {
    uint8_t kind;
    int32_t rows;
    int32_t cols;
    void* data;
    uint8_t* scales;
    float* row_scales;
    float global_scale;
};

struct CExpertBank {
    uint8_t is_nvfp4;
    int32_t n_experts;
    int32_t rows;
    int32_t cols;
    uint8_t* data;
    uint8_t* scales;
    float* expert_scale;
};

struct CControllerWeights {
    void* down_w;
    void* forget_w;
    void* gain_w;
    CLinearWeight down_nvfp4;
    CLinearWeight forget_nvfp4;
    CLinearWeight gain_nvfp4;
};

struct CDsparkLayerWeights {
    uint8_t is_local;
    void* input_norm;
    void* post_norm;
    void* dflare_norm;
    void* q_norm;
    void* k_norm;
    void* k_norm_ctx;
    CLinearWeight w_qkv;
    CLinearWeight w_o;
    CLinearWeight w_gu;
    CLinearWeight w_down;
    CLinearWeight w_kv_ctx;
};

struct CDsparkWeights {
    uint8_t loaded;
    float* fuse_alpha;
    void* mask_embed;
    void* norm_w;
    void* markov_dflare_norm;
    void* markov_w1;
    void* rnn_w_sp;
    void* rnn_w_h;
    void* rnn_w_f;
    void* rnn_bias;
    void* hidden_out_w;
    void* hidden_out_b;
    int32_t draft_vocab_size;
    int32_t* draft_vocab_ids;
    int32_t* inv_draft_vocab_ids;
    CLinearWeight unembed_draft;
    CLinearWeight fold_w;
    float* fold_b;
    CDsparkLayerWeights layers[MTP_LAYERS];
};

static LinearWeight to_lw(const CLinearWeight& c) {
    LinearWeight lw;
    lw.kind = static_cast<WeightKind>(c.kind);
    lw.rows = c.rows;
    lw.cols = c.cols;
    lw.data = c.data;
    lw.scales = c.scales;
    lw.row_scales = c.row_scales;
    lw.global_scale = c.global_scale;
    return lw;
}

static ExpertBank to_eb(const CExpertBank& c) {
    ExpertBank eb;
    eb.is_nvfp4 = (c.is_nvfp4 != 0);
    eb.n_experts = c.n_experts;
    eb.rows = c.rows;
    eb.cols = c.cols;
    eb.data = c.data;
    eb.scales = c.scales;
    eb.expert_scale = c.expert_scale;
    return eb;
}

static ControllerWeights to_cw(const CControllerWeights& c) {
    ControllerWeights cw;
    cw.down_w = static_cast<__nv_bfloat16*>(c.down_w);
    cw.forget_w = static_cast<__nv_bfloat16*>(c.forget_w);
    cw.gain_w = static_cast<__nv_bfloat16*>(c.gain_w);
    cw.down_nvfp4 = to_lw(c.down_nvfp4);
    cw.forget_nvfp4 = to_lw(c.forget_nvfp4);
    cw.gain_nvfp4 = to_lw(c.gain_nvfp4);
    return cw;
}

static DsparkWeights to_dw(const CDsparkWeights& c) {
    DsparkWeights dw;
    dw.loaded = (c.loaded != 0);
    dw.fuse_alpha = c.fuse_alpha;
    dw.mask_embed = static_cast<__nv_bfloat16*>(c.mask_embed);
    dw.norm_w = static_cast<__nv_bfloat16*>(c.norm_w);
    dw.markov_dflare_norm = static_cast<__nv_bfloat16*>(c.markov_dflare_norm);
    dw.markov_w1 = static_cast<__nv_bfloat16*>(c.markov_w1);
    dw.rnn_w_sp = static_cast<__nv_bfloat16*>(c.rnn_w_sp);
    dw.rnn_w_h = static_cast<__nv_bfloat16*>(c.rnn_w_h);
    dw.rnn_w_f = static_cast<__nv_bfloat16*>(c.rnn_w_f);
    dw.rnn_bias = static_cast<__nv_bfloat16*>(c.rnn_bias);
    dw.hidden_out_w = static_cast<__nv_bfloat16*>(c.hidden_out_w);
    dw.hidden_out_b = static_cast<__nv_bfloat16*>(c.hidden_out_b);
    dw.draft_vocab_size = c.draft_vocab_size;
    dw.draft_vocab_ids = c.draft_vocab_ids;
    dw.inv_draft_vocab_ids = c.inv_draft_vocab_ids;
    dw.unembed_draft = to_lw(c.unembed_draft);
    dw.fold_w = to_lw(c.fold_w);
    dw.fold_b = c.fold_b;
    for (int l = 0; l < MTP_LAYERS; ++l) {
        dw.layers[l].is_local = (c.layers[l].is_local != 0);
        dw.layers[l].input_norm = static_cast<__nv_bfloat16*>(c.layers[l].input_norm);
        dw.layers[l].post_norm = static_cast<__nv_bfloat16*>(c.layers[l].post_norm);
        dw.layers[l].dflare_norm = static_cast<__nv_bfloat16*>(c.layers[l].dflare_norm);
        dw.layers[l].q_norm = static_cast<__nv_bfloat16*>(c.layers[l].q_norm);
        dw.layers[l].k_norm = static_cast<__nv_bfloat16*>(c.layers[l].k_norm);
        dw.layers[l].k_norm_ctx = static_cast<__nv_bfloat16*>(c.layers[l].k_norm_ctx);
        dw.layers[l].w_qkv = to_lw(c.layers[l].w_qkv);
        dw.layers[l].w_o = to_lw(c.layers[l].w_o);
        dw.layers[l].w_gu = to_lw(c.layers[l].w_gu);
        dw.layers[l].w_down = to_lw(c.layers[l].w_down);
        dw.layers[l].w_kv_ctx = to_lw(c.layers[l].w_kv_ctx);
    }
    return dw;
}

// --- CUDA Runtime Primitives ---

int32_t squid_cuda_malloc(void** ptr, size_t bytes) { return (int32_t)cudaMalloc(ptr, bytes); }
int32_t squid_cuda_free(void* ptr) { return (int32_t)cudaFree(ptr); }
int32_t squid_cuda_malloc_host_pinned(void** ptr, size_t bytes, uint8_t mapped) {
    return (int32_t)cudaHostAlloc(ptr, bytes, mapped ? cudaHostAllocMapped : cudaHostAllocDefault);
}
int32_t squid_cuda_free_host(void* ptr) { return (int32_t)cudaFreeHost(ptr); }
int32_t squid_cuda_host_get_device_pointer(void** dev_ptr, void* host_ptr) {
    return (int32_t)cudaHostGetDevicePointer(dev_ptr, host_ptr, 0);
}
int32_t squid_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return (int32_t)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
int32_t squid_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return (int32_t)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
int32_t squid_cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    return (int32_t)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
}
int32_t squid_cuda_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    return (int32_t)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
}
int32_t squid_cuda_memcpy_d2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    return (int32_t)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
}
int32_t squid_cuda_memset_async(void* dst, int32_t value, size_t bytes, void* stream) {
    return (int32_t)cudaMemsetAsync(dst, value, bytes, static_cast<cudaStream_t>(stream));
}
int32_t squid_cuda_device_sync() { return (int32_t)cudaDeviceSynchronize(); }
int32_t squid_cuda_stream_create(void** out_stream) {
    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    *out_stream = s;
    return (int32_t)e;
}
int32_t squid_cuda_stream_destroy(void* stream) { return (int32_t)cudaStreamDestroy(static_cast<cudaStream_t>(stream)); }
int32_t squid_cuda_stream_sync(void* stream) { return (int32_t)cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); }
int32_t squid_cuda_event_create(void** out_event, uint8_t disable_timing) {
    cudaEvent_t ev = nullptr;
    cudaError_t e = cudaEventCreateWithFlags(&ev, disable_timing ? cudaEventDisableTiming : cudaEventDefault);
    *out_event = ev;
    return (int32_t)e;
}
int32_t squid_cuda_event_destroy(void* ev) { return (int32_t)cudaEventDestroy(static_cast<cudaEvent_t>(ev)); }
int32_t squid_cuda_event_record(void* ev, void* stream) {
    return (int32_t)cudaEventRecord(static_cast<cudaEvent_t>(ev), static_cast<cudaStream_t>(stream));
}
int32_t squid_cuda_event_sync(void* ev) { return (int32_t)cudaEventSynchronize(static_cast<cudaEvent_t>(ev)); }
int32_t squid_cuda_event_elapsed_ms(float* out_ms, void* ev0, void* ev1) {
    return (int32_t)cudaEventElapsedTime(out_ms, static_cast<cudaEvent_t>(ev0), static_cast<cudaEvent_t>(ev1));
}
int32_t squid_cuda_stream_wait_event(void* stream, void* ev) {
    return (int32_t)cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(ev), 0);
}
int32_t squid_cuda_graph_begin(void* stream) {
    return (int32_t)cudaStreamBeginCapture(static_cast<cudaStream_t>(stream), cudaStreamCaptureModeGlobal);
}
int32_t squid_cuda_graph_end_instantiate(void* stream, void** out_exec) {
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(static_cast<cudaStream_t>(stream), &graph);
    if (e != cudaSuccess) return (int32_t)e;
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    *out_exec = exec;
    return (int32_t)e;
}
int32_t squid_cuda_graph_launch(void* exec, void* stream) {
    return (int32_t)cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), static_cast<cudaStream_t>(stream));
}
int32_t squid_cuda_graph_destroy(void* exec) {
    return (int32_t)cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec));
}

// --- Weight Packing Kernels ---

float squid_pack_bf16_to_nvfp4(
    const void* src_bf16_dev, void* dst_data, uint8_t* dst_scales, float* d_amax_scratch,
    int32_t rows, int32_t cols) {
    CUDA_CHECK(cudaMemset(d_amax_scratch, 0, sizeof(float)));
    size_t n = (size_t)rows * cols;
    int grid = (int)std::min<size_t>((n + 255) / 256, 2048);
    z_bf16_amax_kernel<<<grid, 256>>>(static_cast<const __nv_bfloat16*>(src_bf16_dev), n, d_amax_scratch);
    float h_amax = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_amax, d_amax_scratch, sizeof(float), cudaMemcpyDeviceToHost));
    float gscale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);
    size_t total_blocks = (size_t)rows * (cols / 16);
    z_quant_bf16_to_nvfp4_kernel<<<(int)((total_blocks + 255) / 256), 256>>>(
        static_cast<const __nv_bfloat16*>(src_bf16_dev), static_cast<uint8_t*>(dst_data), dst_scales, rows, cols, gscale);
    return gscale;
}

void squid_pack_bf16_to_fp8(
    const void* src_bf16_dev, void* dst_data, float* dst_row_scales, int32_t rows, int32_t cols) {
    z_quant_bf16_to_fp8_kernel<<<rows, 256>>>(
        static_cast<const __nv_bfloat16*>(src_bf16_dev), static_cast<uint8_t*>(dst_data), dst_row_scales, rows, cols);
}

void squid_convert_mxfp8_to_nvfp4_experts(
    const uint8_t* src_w, const uint8_t* src_s,
    uint8_t* dst_w, uint8_t* dst_s, float* dst_expert_scale,
    int32_t n_experts, int32_t rows, int32_t cols) {
    z_mxfp8_expert_amax_kernel<<<n_experts, 256>>>(src_w, src_s, dst_expert_scale, rows, cols);
    size_t total_blocks = (size_t)n_experts * rows * (cols / 16);
    z_mxfp8_to_nvfp4_expert_kernel<<<(int)((total_blocks + 255) / 256), 256>>>(
        src_w, src_s, dst_expert_scale, dst_w, dst_s, n_experts, rows, cols);
}

void squid_split_dspark_rnn(const void* src_bf16, void* w_sp, void* w_h, void* w_f) {
    z_split_dspark_rnn_kernel<<<768, 256>>>(
        static_cast<const __nv_bfloat16*>(src_bf16),
        static_cast<__nv_bfloat16*>(w_sp), static_cast<__nv_bfloat16*>(w_h), static_cast<__nv_bfloat16*>(w_f));
}

void squid_compute_dspark_fold(
    const void* unembed_bf16, const void* hidden_out_w, const void* hidden_out_b,
    float* fold_b, void* fold_bf16_out, int32_t V, int32_t K, int32_t R) {
    dim3 fb_blk(32, 8);
    z_compute_dspark_fold_bias_kernel<<<(V + 7) / 8, fb_blk>>>(
        static_cast<const __nv_bfloat16*>(unembed_bf16), static_cast<const __nv_bfloat16*>(hidden_out_b), fold_b, V, K);
    z_compute_dspark_fold_w_kernel<<<V, 256>>>(
        static_cast<const __nv_bfloat16*>(unembed_bf16), static_cast<const __nv_bfloat16*>(hidden_out_w),
        static_cast<__nv_bfloat16*>(fold_bf16_out), V, K, R);
}

// --- Inference Kernel Trampolines ---

void squid_launch_embed_norm(const void* embed, const void* embed_norm, const int32_t* tokens, float* out, int32_t B, void* stream) {
    launch_embed_norm(static_cast<const __nv_bfloat16*>(embed), static_cast<const __nv_bfloat16*>(embed_norm),
                      tokens, out, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_rmsnorm(const float* x, const void* weight, float* out, int32_t B, int32_t width, int32_t group_size, float eps, void* stream) {
    launch_rmsnorm(x, static_cast<const __nv_bfloat16*>(weight), out, B, width, group_size, eps, static_cast<cudaStream_t>(stream));
}

void squid_launch_extract_block(const float* blocks, int32_t max_blocks, int32_t slot_idx, float* dst, int32_t B, void* stream) {
    launch_extract_block(blocks, max_blocks, slot_idx, dst, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_fused_depth_norm(
    const float* hidden, float* partial_out, const float* blocks, int32_t n_blocks, const float* queries,
    const void* n0, const void* n1, const void* n2,
    float* o0, float* o1, float* o2, void* oh0, void* oh1, void* oh2, int32_t n_out, int32_t B, void* stream) {
    launch_fused_depth_norm(
        hidden, partial_out, blocks, n_blocks, queries,
        static_cast<const __nv_bfloat16*>(n0), static_cast<const __nv_bfloat16*>(n1), static_cast<const __nv_bfloat16*>(n2),
        o0, o1, o2, static_cast<__half*>(oh0), static_cast<__half*>(oh1), static_cast<__half*>(oh2),
        n_out, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_controller_mega_nvfp4(
    float* const* xs, const float* saved, const float* prior_last, const float* const* cp_queries,
    const CControllerWeights* const* cws_in, const void* const* post_norms,
    float* const* outs, void* const* outs_h, int32_t n_streams, int32_t B, void* stream) {
    ControllerWeights tmp[3];
    const ControllerWeights* cws[3] = {nullptr, nullptr, nullptr};
    const float* q3[3] = {nullptr, nullptr, nullptr};
    const __nv_bfloat16* pn3[3] = {nullptr, nullptr, nullptr};
    __half* oh3[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < n_streams; ++i) {
        tmp[i] = to_cw(*cws_in[i]);
        cws[i] = &tmp[i];
        q3[i] = cp_queries[i];
        pn3[i] = static_cast<const __nv_bfloat16*>(post_norms[i]);
        oh3[i] = outs_h ? static_cast<__half*>(outs_h[i]) : nullptr;
    }
    launch_controller_mega_nvfp4(xs, saved, prior_last, q3, cws, pn3, outs, outs_h ? oh3 : nullptr,
                                 n_streams, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_controller_merge_batched(
    float* const* xs, const float* saved, const float* prior_last, const float* const* cp_queries,
    const CControllerWeights* const* cws_in, const void* const* post_norms,
    float* const* outs, void* const* outs_h, int32_t n_streams,
    float* scratch_msg, float* scratch_cat, float* scratch_d, int32_t B, void* stream) {
    ControllerWeights tmp[3];
    const ControllerWeights* cws[3] = {nullptr, nullptr, nullptr};
    const float* q3[3] = {nullptr, nullptr, nullptr};
    const __nv_bfloat16* pn3[3] = {nullptr, nullptr, nullptr};
    __half* oh3[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < n_streams; ++i) {
        tmp[i] = to_cw(*cws_in[i]);
        cws[i] = &tmp[i];
        q3[i] = cp_queries[i];
        pn3[i] = static_cast<const __nv_bfloat16*>(post_norms[i]);
        oh3[i] = outs_h ? static_cast<__half*>(outs_h[i]) : nullptr;
    }
    launch_controller_merge_batched(xs, saved, prior_last, q3, cws, pn3, outs, outs_h ? oh3 : nullptr,
                                    n_streams, scratch_msg, scratch_cat, scratch_d, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_qkvrg_fused(
    const CLinearWeight* w_qrgs, const CLinearWeight* w_k, const CLinearWeight* w_v,
    const void* in_qh, const void* in_kh, const void* in_vh,
    float* qrgs_out, float* k_out, float* v_out, int32_t B, void* stream) {
    LayerWeights lw;
    lw.w_qrgs = to_lw(*w_qrgs);
    lw.w_k = to_lw(*w_k);
    lw.w_v = to_lw(*w_v);
    launch_qkvrg_fused(lw, static_cast<const __half*>(in_qh), static_cast<const __half*>(in_kh),
                       static_cast<const __half*>(in_vh), qrgs_out, k_out, v_out, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_linear(const CLinearWeight* w, const float* x, float* y, int32_t B, uint8_t add_res, void* stream) {
    launch_linear(to_lw(*w), x, y, B, add_res != 0, static_cast<cudaStream_t>(stream));
}

void squid_launch_linear_h(const CLinearWeight* w, const void* x_h, float* y, int32_t B, uint8_t add_res, void* stream) {
    launch_linear_h(to_lw(*w), static_cast<const __half*>(x_h), y, B, add_res != 0, static_cast<cudaStream_t>(stream));
}

void squid_launch_mega_global_sconv_prep_store(
    float* qrgs, const float* k_proj, const float* v_proj, const float* in_k, const float* in_v,
    const void* k_sw, const void* v_sw, const void* k_sp, const void* v_sp,
    float* k_ring, float* v_ring, const void* q_cs, const void* r_cs,
    void* k_cache, void* v_cache, void* k_scale, void* v_scale, uint8_t kv_fmt,
    int32_t kv_cap, int32_t B, const int32_t* d_pos, void* stream) {
    launch_mega_global_sconv_prep_store(
        qrgs, k_proj, v_proj, in_k, in_v,
        static_cast<const __nv_bfloat16*>(k_sw), static_cast<const __nv_bfloat16*>(v_sw),
        static_cast<const __nv_bfloat16*>(k_sp), static_cast<const __nv_bfloat16*>(v_sp),
        k_ring, v_ring, static_cast<const __nv_bfloat16*>(q_cs), static_cast<const __nv_bfloat16*>(r_cs),
        k_cache, v_cache, static_cast<__half*>(k_scale), static_cast<__half*>(v_scale),
        static_cast<KVFormat>(kv_fmt), kv_cap, B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_mega_local_sconv_prep_store(
    float* qrgs, const float* k_proj, const float* v_proj, const void* k_sw, const void* v_sw,
    float* k_ring, float* v_ring, const void* q_cs, const void* r_cs,
    void* k_cache, void* v_cache, void* k_scale, void* v_scale, uint8_t kv_fmt,
    int32_t kv_cap, int32_t B, const int32_t* d_pos, void* stream) {
    launch_mega_local_sconv_prep_store(
        qrgs, k_proj, v_proj, static_cast<const __nv_bfloat16*>(k_sw), static_cast<const __nv_bfloat16*>(v_sw),
        k_ring, v_ring, static_cast<const __nv_bfloat16*>(q_cs), static_cast<const __nv_bfloat16*>(r_cs),
        k_cache, v_cache, static_cast<__half*>(k_scale), static_cast<__half*>(v_scale),
        static_cast<KVFormat>(kv_fmt), kv_cap, B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_relative_attention_gated(
    const float* qrgs, const void* rel_proj, const void* k_cache, const void* v_cache,
    const void* k_scale, const void* v_scale, uint8_t kv_fmt, float* split_buf,
    float* out_gated, void* out_gated_h, int32_t kv_cap, int32_t window, int32_t B, const int32_t* d_pos, void* stream) {
    launch_relative_attention_gated(
        qrgs, static_cast<const __nv_bfloat16*>(rel_proj), k_cache, v_cache,
        static_cast<const __half*>(k_scale), static_cast<const __half*>(v_scale),
        static_cast<KVFormat>(kv_fmt), split_buf, out_gated, static_cast<__half*>(out_gated_h),
        kv_cap, window, B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_mega_attn_post_mlp_in(
    const float* o_proj, const void* attn_sw, float* attn_ring, const float* hidden, const float* partial_prev,
    float* blocks, int32_t block_count_before, uint8_t is_boundary, float* residual_out, const float* mlp_query,
    const void* mlp_norm_w, float* mlp_out, void* mlp_out_h, int32_t B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, void* stream) {
    launch_mega_attn_post_mlp_in(
        o_proj, static_cast<const __nv_bfloat16*>(attn_sw), attn_ring, hidden, partial_prev,
        blocks, block_count_before, is_boundary != 0, residual_out, mlp_query,
        static_cast<const __nv_bfloat16*>(mlp_norm_w), mlp_out, static_cast<__half*>(mlp_out_h),
        B, d_pos, abl_r, abl_d, abl_p, static_cast<cudaStream_t>(stream));
}

void squid_launch_route_topk_nvfp4(
    const void* m_h, const CLinearWeight* router_nvfp4, const float* router_bias,
    float* scratch_logits, int32_t* out_ids, float* out_weights, void* groups, int32_t B, void* stream) {
    launch_route_topk_nvfp4(
        static_cast<const __half*>(m_h), to_lw(*router_nvfp4), router_bias,
        scratch_logits, out_ids, out_weights, static_cast<ExpertGroups*>(groups), nullptr, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_route_topk(
    const float* m, const void* router_w, const float* router_bias,
    float* scratch_logits, int32_t* out_ids, float* out_weights, void* groups, int32_t B, void* stream) {
    launch_route_topk(
        m, static_cast<const __nv_bfloat16*>(router_w), router_bias,
        scratch_logits, out_ids, out_weights, static_cast<ExpertGroups*>(groups), nullptr, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_moe_gate_up_h(
    const CExpertBank* w13, const void* x_h, const int32_t* route_ids, const void* groups, void* out_act_h, int32_t B, void* stream) {
    launch_moe_gate_up_h(to_eb(*w13), static_cast<const __half*>(x_h), route_ids,
                         static_cast<const ExpertGroups*>(groups), static_cast<__half*>(out_act_h), B, static_cast<cudaStream_t>(stream));
}

void squid_launch_moe_down_h(
    const CExpertBank* w2, const void* act_h, const int32_t* route_ids, const float* route_w,
    const void* groups, float* out_down, int32_t B, void* stream) {
    launch_moe_down_h(to_eb(*w2), static_cast<const __half*>(act_h), route_ids, route_w,
                      static_cast<const ExpertGroups*>(groups), out_down, B, static_cast<cudaStream_t>(stream), true);
}

void squid_launch_mega_mlp_post_res(
    const float* moe_down_or_slots, uint8_t moe_is_slotted, const void* mlp_sw, float* mlp_ring,
    const float* residual, float* hidden_out, int32_t B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, void* stream) {
    launch_mega_mlp_post_res(
        moe_down_or_slots, moe_is_slotted != 0, static_cast<const __nv_bfloat16*>(mlp_sw),
        mlp_ring, residual, hidden_out, B, d_pos, abl_r, abl_d, abl_p, static_cast<cudaStream_t>(stream));
}

void squid_launch_mega_mlp_post_next_depth_norm(
    const float* moe_down_or_slots, uint8_t moe_is_slotted, const void* prev_mlp_sw, float* prev_mlp_ring,
    const float* residual, float* hidden_out, float* pre_tap_out, float* post_tap_out, float* partial_out,
    const float* blocks, int32_t n_blocks, const float* queries,
    const void* n0, const void* n1, const void* n2,
    float* o0, float* o1, float* o2, void* oh0, void* oh1, void* oh2,
    int32_t n_out, int32_t B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, void* stream) {
    launch_mega_mlp_post_next_depth_norm(
        moe_down_or_slots, moe_is_slotted != 0, static_cast<const __nv_bfloat16*>(prev_mlp_sw),
        prev_mlp_ring, residual, hidden_out, pre_tap_out, post_tap_out, partial_out,
        blocks, n_blocks, queries,
        static_cast<const __nv_bfloat16*>(n0), static_cast<const __nv_bfloat16*>(n1), static_cast<const __nv_bfloat16*>(n2),
        o0, o1, o2, static_cast<__half*>(oh0), static_cast<__half*>(oh1), static_cast<__half*>(oh2),
        n_out, B, d_pos, abl_r, abl_d, abl_p, static_cast<cudaStream_t>(stream));
}

void squid_launch_lm_head_argmax(
    const CLinearWeight* unembed, const float* x, int32_t* out_tokens, float* out_logits,
    float* scratch_vals, int32_t* scratch_idxs, const SampleConfig* sample_cfg, const int32_t* d_pos,
    const int32_t* d_draft_tokens, const float* d_draft_probs, const float* d_draft_q_sel,
    const int32_t* d_draft_is_delta, const int32_t* inv_draft_vocab, int32_t draft_vocab_size,
    int32_t B, void* stream) {
    launch_lm_head_argmax(to_lw(*unembed), x, out_tokens, out_logits, scratch_vals, scratch_idxs,
                          sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel,
                          d_draft_is_delta, inv_draft_vocab, draft_vocab_size, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_dspark_commit(
    const CDsparkWeights* dw_in, const float* taps,
    void* const* ctx_k, void* const* ctx_v, void* const* ctx_ks, void* const* ctx_vs,
    const int32_t* ctx_cap, uint8_t global_kv_fmt,
    float* f_ring, float* scratch_mix, float* scratch_kv, int32_t B, const int32_t* d_pos, void* stream) {
    DsparkWeights dw = to_dw(*dw_in);
    void* ck[MTP_LAYERS] = {ctx_k[0], ctx_k[1], ctx_k[2]};
    void* cv[MTP_LAYERS] = {ctx_v[0], ctx_v[1], ctx_v[2]};
    __half* cks[MTP_LAYERS] = {static_cast<__half*>(ctx_ks[0]), static_cast<__half*>(ctx_ks[1]), static_cast<__half*>(ctx_ks[2])};
    __half* cvs[MTP_LAYERS] = {static_cast<__half*>(ctx_vs[0]), static_cast<__half*>(ctx_vs[1]), static_cast<__half*>(ctx_vs[2])};
    int cap[MTP_LAYERS] = {ctx_cap[0], ctx_cap[1], ctx_cap[2]};
    launch_dspark_commit(dw, taps, ck, cv, cks, cvs, cap, static_cast<KVFormat>(global_kv_fmt),
                         f_ring, scratch_mix, scratch_kv, B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_dspark_draft(
    const void* embed, const void* embed_norm, const CDsparkWeights* dw_in,
    const int32_t* d_anchor_token, const int32_t* d_pos, int32_t num_drafts,
    void* const* ctx_k, void* const* ctx_v, const void* const* ctx_ks, const void* const* ctx_vs,
    const int32_t* ctx_cap, uint8_t global_kv_fmt, float* split_buf,
    const float* f_ring, float* workspace, int32_t* out_draft_tokens, float* out_draft_conf,
    const SampleConfig* sample_cfg, float* out_draft_probs, float* out_draft_q_sel,
    int32_t* out_draft_is_delta, void* stream) {
    ModelWeights mw;
    mw.embed = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(embed));
    mw.embed_norm = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(embed_norm));
    mw.dspark = to_dw(*dw_in);
    void* ck[MTP_LAYERS] = {ctx_k[0], ctx_k[1], ctx_k[2]};
    void* cv[MTP_LAYERS] = {ctx_v[0], ctx_v[1], ctx_v[2]};
    const __half* cks[MTP_LAYERS] = {static_cast<const __half*>(ctx_ks[0]), static_cast<const __half*>(ctx_ks[1]), static_cast<const __half*>(ctx_ks[2])};
    const __half* cvs[MTP_LAYERS] = {static_cast<const __half*>(ctx_vs[0]), static_cast<const __half*>(ctx_vs[1]), static_cast<const __half*>(ctx_vs[2])};
    int cap[MTP_LAYERS] = {ctx_cap[0], ctx_cap[1], ctx_cap[2]};
    launch_dspark_draft(mw, d_anchor_token, d_pos, num_drafts, ck, cv, cks, cvs, cap,
                        static_cast<KVFormat>(global_kv_fmt), split_buf, f_ring, workspace,
                        out_draft_tokens, out_draft_conf, sample_cfg, out_draft_probs,
                        out_draft_q_sel, out_draft_is_delta, static_cast<cudaStream_t>(stream));
}

int32_t squid_adaptive_graph_create(
    const int32_t* d_anchor, const int32_t* d_drafts, const float* d_draft_conf, float threshold,
    int32_t* d_tokens, int32_t* d_chosen_width,
    const int32_t* d_accepted, int32_t* h_accepted,
    const int32_t* d_num_acc, int32_t* h_num_acc, int32_t* h_chosen_width,
    void* stream_raw, void** out_graph) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_raw);
    CAdaptiveGraph* ag = new CAdaptiveGraph();
    cudaError_t err = cudaGraphCreate(&ag->root_graph, 0);
    if (err != cudaSuccess) { delete ag; return (int32_t)err; }
    err = cudaGraphConditionalHandleCreate(&ag->handle, ag->root_graph, 0, 0);
    if (err != cudaSuccess) { cudaGraphDestroy(ag->root_graph); delete ag; return (int32_t)err; }

    err = cudaStreamBeginCaptureToGraph(s, ag->root_graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) { cudaGraphDestroy(ag->root_graph); delete ag; return (int32_t)err; }

    z_adaptive_select_width_kernel<<<1, 1, 0, s>>>(
        d_anchor, d_drafts, d_draft_conf, threshold, d_tokens, d_chosen_width, ag->handle);

    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    size_t ndeps = 0;
    err = cudaStreamGetCaptureInfo(s, &status, nullptr, &g, &deps, &edges, &ndeps);
    if (err != cudaSuccess) {
        cudaGraph_t disc = nullptr; cudaStreamEndCapture(s, &disc);
        cudaGraphDestroy(ag->root_graph); delete ag; return (int32_t)err;
    }

    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = ag->handle;
    p.conditional.type = cudaGraphCondTypeSwitch;
    p.conditional.size = 3;
    cudaGraphNode_t sw_node = nullptr;
    err = cudaGraphAddNode(&sw_node, g, deps, edges, ndeps, &p);
    if (err != cudaSuccess) {
        cudaGraph_t disc = nullptr; cudaStreamEndCapture(s, &disc);
        cudaGraphDestroy(ag->root_graph); delete ag; return (int32_t)err;
    }
    ag->bodies[0] = p.conditional.phGraph_out[0];
    ag->bodies[1] = p.conditional.phGraph_out[1];
    ag->bodies[2] = p.conditional.phGraph_out[2];

    err = cudaStreamUpdateCaptureDependencies(s, &sw_node, nullptr, 1, cudaStreamSetCaptureDependencies);
    if (err != cudaSuccess) {
        cudaGraph_t disc = nullptr; cudaStreamEndCapture(s, &disc);
        cudaGraphDestroy(ag->root_graph); delete ag; return (int32_t)err;
    }

    // Async D2H copy of verified outputs right after the switch node finishes!
    cudaMemcpyAsync(h_accepted, d_accepted, MTP_BLOCK_SIZE * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_num_acc, d_num_acc, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_chosen_width, d_chosen_width, sizeof(int32_t), cudaMemcpyDeviceToHost, s);

    cudaGraph_t out_g = nullptr;
    err = cudaStreamEndCapture(s, &out_g);
    if (err != cudaSuccess) { cudaGraphDestroy(ag->root_graph); delete ag; return (int32_t)err; }
    *out_graph = ag;
    return 0;
}

int32_t squid_adaptive_graph_branch_begin(void* graph_raw, int32_t branch_idx, void* stream_raw) {
    CAdaptiveGraph* ag = static_cast<CAdaptiveGraph*>(graph_raw);
    return (int32_t)cudaStreamBeginCaptureToGraph(
        static_cast<cudaStream_t>(stream_raw), ag->bodies[branch_idx],
        nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
}

int32_t squid_adaptive_graph_branch_end(void* stream_raw) {
    cudaGraph_t out_g = nullptr;
    return (int32_t)cudaStreamEndCapture(static_cast<cudaStream_t>(stream_raw), &out_g);
}

int32_t squid_adaptive_graph_instantiate(void* graph_raw) {
    CAdaptiveGraph* ag = static_cast<CAdaptiveGraph*>(graph_raw);
    cudaError_t err = cudaGraphInstantiate(&ag->exec, ag->root_graph, 0);
    if (ag->root_graph) {
        cudaGraphDestroy(ag->root_graph);
        ag->root_graph = nullptr;
    }
    return (int32_t)err;
}

int32_t squid_adaptive_graph_launch(void* graph_raw, void* stream_raw) {
    CAdaptiveGraph* ag = static_cast<CAdaptiveGraph*>(graph_raw);
    return (int32_t)cudaGraphLaunch(ag->exec, static_cast<cudaStream_t>(stream_raw));
}

int32_t squid_adaptive_graph_destroy(void* graph_raw) {
    if (!graph_raw) return 0;
    CAdaptiveGraph* ag = static_cast<CAdaptiveGraph*>(graph_raw);
    if (ag->exec) cudaGraphExecDestroy(ag->exec);
    if (ag->root_graph) cudaGraphDestroy(ag->root_graph);
    delete ag;
    return 0;
}

void squid_launch_spec_verify(
    const int32_t* target_preds, const int32_t* draft_tokens, int32_t B,
    int32_t* out_accepted, int32_t* out_num_accepted, int32_t* d_pos, int32_t* d_next_anchor, void* stream) {
    launch_spec_verify(target_preds, draft_tokens, B, out_accepted, out_num_accepted, d_pos, d_next_anchor,
                       static_cast<cudaStream_t>(stream));
}

void squid_launch_inc_pos_and_copy_tok(int32_t* d_pos, int32_t* d_tok_in, const int32_t* d_tok_out, void* stream) {
    z_inc_pos_and_copy_tok_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(d_pos, d_tok_in, d_tok_out);
}

void squid_launch_inc_pos_by(int32_t* d_pos, int32_t delta, void* stream) {
    z_inc_pos_by_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(d_pos, delta);
}

void squid_launch_assemble_verify_tokens(const int32_t* d_anchor, const int32_t* d_drafts, int32_t* d_tokens, int32_t num_drafts, void* stream) {
    z_assemble_verify_tokens_kernel<<<1, num_drafts + 1, 0, static_cast<cudaStream_t>(stream)>>>(d_anchor, d_drafts, d_tokens, num_drafts);
}

size_t squid_sizeof_expert_groups() { return sizeof(ExpertGroups); }

void squid_launch_prefill_warmup_cublas(void* stream) {
    launch_prefill_warmup_cublas(static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_f32_to_f16(const float* src, void* dst_h, int32_t count, void* stream) {
    launch_prefill_f32_to_f16(src, static_cast<__half*>(dst_h), count, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_gemm_h(
    const CLinearWeight* w, const void* x_h, float* y, void* w_f16_scratch, int32_t B, void* stream) {
    launch_prefill_gemm_h(
        to_lw(*w), static_cast<const __half*>(x_h), y, static_cast<__half*>(w_f16_scratch),
        B, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_sconv_post_res(
    const float* src_or_slots, int32_t is_slotted, const void* sconv_w, float* ring,
    const float* residual, float* out_res, float* pre_tap_out,
    float* blocks, const float* partial, int32_t block_idx, int32_t is_boundary,
    int32_t B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, void* stream) {
    launch_prefill_sconv_post_res(
        src_or_slots, is_slotted, static_cast<const __nv_bfloat16*>(sconv_w), ring,
        residual, out_res, pre_tap_out, blocks, partial, block_idx, is_boundary,
        B, d_pos, abl_r, abl_d, abl_p, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_copy_taps(const float* src, float* dst_taps, int32_t tap_slot, int32_t B, void* stream) {
    launch_prefill_copy_taps(src, dst_taps, tap_slot, B, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_local_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const void* k_sw, const void* v_sw,
    float* k_ring, float* v_ring,
    const void* q_cs, const void* r_cs,
    void* k_cache, void* v_cache, void* k_scale, void* v_scale, uint8_t kv_fmt,
    int32_t kv_cap, int32_t B, const int32_t* d_pos, void* stream) {
    launch_prefill_local_sconv_prep_store(
        qrgs, k_proj, v_proj,
        static_cast<const __nv_bfloat16*>(k_sw), static_cast<const __nv_bfloat16*>(v_sw),
        k_ring, v_ring,
        static_cast<const __nv_bfloat16*>(q_cs), static_cast<const __nv_bfloat16*>(r_cs),
        k_cache, v_cache, static_cast<__half*>(k_scale), static_cast<__half*>(v_scale),
        static_cast<KVFormat>(kv_fmt), kv_cap, B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_global_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const void* in_k_h, const void* in_v_h,
    const void* k_sw, const void* v_sw,
    const void* k_sp, const void* v_sp,
    float* k_ring, float* v_ring,
    const void* q_cs, const void* r_cs,
    void* k_cache, void* v_cache, void* k_scale, void* v_scale,
    uint8_t kv_fmt, int32_t kv_cap,
    float* scratch_taps, void* w_f16_scratch,
    int32_t B, const int32_t* d_pos, void* stream) {
    launch_prefill_global_sconv_prep_store(
        qrgs, k_proj, v_proj,
        static_cast<const __half*>(in_k_h), static_cast<const __half*>(in_v_h),
        static_cast<const __nv_bfloat16*>(k_sw), static_cast<const __nv_bfloat16*>(v_sw),
        static_cast<const __nv_bfloat16*>(k_sp), static_cast<const __nv_bfloat16*>(v_sp),
        k_ring, v_ring,
        static_cast<const __nv_bfloat16*>(q_cs), static_cast<const __nv_bfloat16*>(r_cs),
        k_cache, v_cache, static_cast<__half*>(k_scale), static_cast<__half*>(v_scale),
        static_cast<KVFormat>(kv_fmt), kv_cap,
        scratch_taps, static_cast<__half*>(w_f16_scratch),
        B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_flash_attn_tc(
    const float* qrgs, const void* rel_proj,
    const void* k_cache, const void* v_cache,
    const void* k_scale, const void* v_scale,
    uint8_t kv_fmt, void* out_gated_h,
    int32_t kv_cap, int32_t window, int32_t B, const int32_t* d_pos, void* stream) {
    launch_prefill_flash_attn_tc(
        qrgs, static_cast<const __nv_bfloat16*>(rel_proj),
        k_cache, v_cache, static_cast<const __half*>(k_scale), static_cast<const __half*>(v_scale),
        static_cast<KVFormat>(kv_fmt), static_cast<__half*>(out_gated_h),
        kv_cap, window, B, d_pos, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_controller_one(
    const float* x, const float* saved, const float* prior_last, const float* cp_q,
    const CControllerWeights* cw_in, const void* post_norm,
    float* out, void* out_h,
    float* scratch_msg, void* scratch_cat_h, float* scratch_d, void* scratch_d_h,
    float* scratch_f, float* scratch_g, void* w_f16_scratch,
    int32_t B, void* stream) {
    launch_prefill_controller_one(
        x, saved, prior_last, cp_q, to_cw(*cw_in), static_cast<const __nv_bfloat16*>(post_norm),
        out, static_cast<__half*>(out_h),
        scratch_msg, static_cast<__half*>(scratch_cat_h), scratch_d, static_cast<__half*>(scratch_d_h),
        scratch_f, scratch_g, static_cast<__half*>(w_f16_scratch),
        B, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_route_and_sort(
    const float* router_logits, const float* router_bias, int32_t* route_ids, float* route_w,
    int32_t* expert_offsets, int32_t* sorted_tok, int32_t* sorted_slot, float* sorted_w,
    int32_t B, void* stream) {
    launch_prefill_route_and_sort(
        router_logits, router_bias, route_ids, route_w,
        expert_offsets, sorted_tok, sorted_slot, sorted_w,
        B, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_moe_gate_up_h(
    const CExpertBank* w13, const void* x_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    void* out_act_h, int32_t B, void* stream) {
    launch_prefill_moe_gate_up_h(
        to_eb(*w13), static_cast<const __half*>(x_h),
        expert_offsets, sorted_tok, sorted_slot,
        static_cast<__half*>(out_act_h), B, static_cast<cudaStream_t>(stream));
}

void squid_launch_prefill_moe_down_h(
    const CExpertBank* w2, const void* act_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    const float* sorted_w, float* out_slots, int32_t B, void* stream) {
    launch_prefill_moe_down_h(
        to_eb(*w2), static_cast<const __half*>(act_h),
        expert_offsets, sorted_tok, sorted_slot, sorted_w,
        out_slots, B, static_cast<cudaStream_t>(stream));
}

} // extern "C"

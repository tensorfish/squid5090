#include "kernels.hpp"
#include "kernels_common.cuh"

#include <cstdio>

namespace squid {

namespace {

// ============================================================================
// Fast F32 -> F16 conversion (vectorized float4 -> half2x2)
// ============================================================================
__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(src + i);
        __half2 h0 = __floats2half2_rn(v.x, v.y);
        __half2 h1 = __floats2half2_rn(v.z, v.w);
        *reinterpret_cast<__half2*>(dst + i) = h0;
        *reinterpret_cast<__half2*>(dst + i + 2) = h1;
    } else {
        for (int k = i; k < n && k < i + 4; ++k) dst[k] = __float2half(src[k]);
    }
}

// ============================================================================
// ZERO-SMEM L1-CACHED NVFP4 / FP8 / BF16 Linear Kernels (B = 1..8)
// Uses Blackwell sm_120a hardware PTX cvt.rn.f16x2.e2m1x2, 0 bytes SMEM, 0 spills,
// 48 warps/SM max occupancy, and reads small activations directly from L1 cache!
// ============================================================================

template <int B_CONST, bool ADD_RES, int TPR = 32>
__global__ void __launch_bounds__(128)
linear_nvfp4_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const __half* __restrict__ x_h, // [B, K]
    float* __restrict__ y,          // [B, N]
    int N, int K) {
    static_assert(TPR == 16 || TPR == 32);
    constexpr int ROWS_PER_BLK = 128 / TPR;
    int r_in_blk = (TPR == 32) ? (threadIdx.x >> 5) : (threadIdx.x >> 4);
    int lane = threadIdx.x & (TPR - 1);
    int row = blockIdx.x * ROWS_PER_BLK + r_in_blk;
    if (row >= N) return;

    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (K >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * (K >> 4);
    int n_blocks16 = K >> 4;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int blk = lane; blk < n_blocks16; blk += TPR) {
        uint2 w8 = __ldg(w_row + blk);
        float sc = cvt_e4m3_to_float(__ldg(s_row + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);

        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * K + koff);
            uint4 xv0 = __ldg(x4);
            uint4 xv1 = __ldg(x4 + 1);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            __half2 p0 = __hmul2(wh[0], xb0[0]);
            __half2 p1 = __hmul2(wh[1], xb0[1]);
            p0 = __hfma2(wh[2], xb0[2], p0);
            p1 = __hfma2(wh[3], xb0[3], p1);
            p0 = __hfma2(wh[4], xb1[0], p0);
            p1 = __hfma2(wh[5], xb1[1], p1);
            p0 = __hfma2(wh[6], xb1[2], p0);
            p1 = __hfma2(wh[7], xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }

    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = ((TPR == 32) ? warp_sum(acc[b]) : half_warp_sum(acc[b])) * global_scale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

// ============================================================================
// Register-Permuted Blackwell Tensor Core (mma.m16n8k16) Helpers for NVFP4 & MXFP8
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0).
// ============================================================================

struct Nvfp4Chunk32 {
    uint4 q;
    uint16_t s;
};

struct Mxfp8Chunk32 {
    uint4 q0, q1;
    uint8_t s;
};

struct Fp8Chunk32 {
    uint4 q0, q1;
};

__device__ __forceinline__ Fp8Chunk32 load_fp8_chunk32(const uint8_t* codes, int c) {
    Fp8Chunk32 ch;
    ch.q0 = ld_stream_u4(codes + (size_t)c * 32);
    ch.q1 = ld_stream_u4(codes + (size_t)c * 32 + 16);
    return ch;
}

// Lossless E4M3 -> FP16 unpack (every E4M3 finite value is bit-exact in FP16) for mma.m16n8k16.f32.f16!
__device__ __forceinline__ void decode_fp8_chunk32_f16(const Fp8Chunk32& ch, uint32_t (&o)[16]) {
    const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        __half2 h0 = cvt_e4m3x2_to_half2(uint16_t(words[w] & 0xFFFFu));
        __half2 h1 = cvt_e4m3x2_to_half2(uint16_t(words[w] >> 16));
        o[2 * w + 0] = *reinterpret_cast<uint32_t*>(&h0);
        o[2 * w + 1] = *reinterpret_cast<uint32_t*>(&h1);
    }
}

__device__ __forceinline__ Nvfp4Chunk32 load_nvfp4_chunk32(
    const uint8_t* codes, const uint8_t* scales, int c) {
    Nvfp4Chunk32 ch;
    ch.q = ld_stream_u4(codes + (size_t)c * 16);
    ch.s = ld_scale_u16(scales + (size_t)c * 2);
    return ch;
}

__device__ __forceinline__ Mxfp8Chunk32 load_mxfp8_chunk32(
    const uint8_t* codes, const uint8_t* scales, int c) {
    Mxfp8Chunk32 ch;
    ch.q0 = ld_stream_u4(codes + (size_t)c * 32);
    ch.q1 = ld_stream_u4(codes + (size_t)c * 32 + 16);
    ch.s = ld_scale_u8(scales + c);
    return ch;
}

// Exact FP16 decode of 32 NVFP4 weights * 2 E4M3 scales -> 16 half2 words!
// Range: |e2m1 * e4m3| in {0} U [2^-10, 2688], <= 6 mantissa bits -> 100% bit-exact in FP16!
__device__ __forceinline__ void decode_nvfp4_chunk32_f16(const Nvfp4Chunk32& ch, uint32_t (&o)[16]) {
    const __half2 sc = cvt_e4m3x2_to_half2(ch.s);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const uint32_t words[4] = {ch.q.x, ch.q.y, ch.q.z, ch.q.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        __half2 h0, h1, h2, h3;
        cvt_e2m1x8_to_half2x4(words[w], h0, h1, h2, h3);
        const __half2 sm = (w < 2) ? slo : shi;
        __half2 p0 = __hmul2(h0, sm), p1 = __hmul2(h1, sm), p2 = __hmul2(h2, sm), p3 = __hmul2(h3, sm);
        o[4 * w + 0] = *reinterpret_cast<uint32_t*>(&p0);
        o[4 * w + 1] = *reinterpret_cast<uint32_t*>(&p1);
        o[4 * w + 2] = *reinterpret_cast<uint32_t*>(&p2);
        o[4 * w + 3] = *reinterpret_cast<uint32_t*>(&p3);
    }
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void decode_mxfp8_chunk32_bf16(const Mxfp8Chunk32& ch, uint32_t (&o)[16]) {
    const float sc = cvt_ue8m0_to_float(ch.s);
    const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            __half2 h2 = cvt_e4m3x2_to_half2(uint16_t(words[w] >> (16 * hh)));
            float2 v = __half22float2(h2);
            o[2 * w + hh] = pack_bf16x2(v.x * sc, v.y * sc);
        }
    }
}

// ============================================================================
// PDL-Prefetched B=1 + Tensor Core (mma.m16n8k16) B=2..8 Dense NVFP4 GEMVs
// ============================================================================

// B=1 PDL-prefetched NVFP4 GEMV (16 threads/row, ITERS = K / 256):
// Issues ALL weight & scale streaming loads BEFORE pdl_wait_fenced(zero) -> overlaps predecessor tail!
template <bool ADD_RES, int ITERS>
__global__ void __launch_bounds__(128)
linear_nvfp4_b1_pdl_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const __half* __restrict__ x_h,
    float* __restrict__ y,
    int N, int K, uint32_t zero) {
    int hw = threadIdx.x >> 4;   // 0..7
    int lane = threadIdx.x & 15; // 0..15
    int row = blockIdx.x * 8 + hw;
    if (row >= N) return;

    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (K >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * (K >> 4);

    uint2 w8[ITERS];
    uint8_t s8[ITERS];
    #pragma unroll
    for (int iter = 0; iter < ITERS; ++iter) {
        int blk = lane + iter * 16;
        w8[iter] = ld_stream_u2(w_row + blk);
        s8[iter] = ld_scale_u8(s_row + blk);
    }
    pdl_wait_fenced(zero);
    pdl_trigger();

    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < ITERS; ++iter) {
        int koff = (lane + iter * 16) << 4;
        uint4 xv0 = ld_dep_u4(x_h + koff);
        uint4 xv1 = ld_dep_u4(x_h + koff + 8);
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8[iter].x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8[iter].y, wh[4], wh[5], wh[6], wh[7]);
        float sc = cvt_e4m3_to_float(s8[iter]);
        const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
        __half2 p0 = __hmul2(wh[0], xb0[0]);
        __half2 p1 = __hmul2(wh[1], xb0[1]);
        p0 = __hfma2(wh[2], xb0[2], p0);
        p1 = __hfma2(wh[3], xb0[3], p1);
        p0 = __hfma2(wh[4], xb1[0], p0);
        p1 = __hfma2(wh[5], xb1[1], p1);
        p0 = __hfma2(wh[6], xb1[2], p0);
        p1 = __hfma2(wh[7], xb1[3], p1);
        float2 pf = __half22float2(__hadd2(p0, p1));
        acc = fmaf(pf.x + pf.y, sc, acc);
    }

    float s = half_warp_sum(acc) * global_scale;
    if (lane == 0) {
        if constexpr (ADD_RES) y[row] += s;
        else                   y[row] = s;
    }
}

// B=2..8 Tensor Core NVFP4 GEMV (16 rows/CTA, SPLITK warps, loops in batches of U=3 steps of 128 cols/warp).
// Pre-issues batch 0's weight chunks BEFORE pdl_wait_fenced(zero) + binds each batch with dep = xor & zero!
template <int B_CONST, bool ADD_RES, int K_DIM, int SPLITK>
__global__ void __launch_bounds__(SPLITK * 32)
linear_nvfp4_mma_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const __half* __restrict__ x_h,
    float* __restrict__ y,
    int N, uint32_t zero) {
    constexpr int TOTAL_U = K_DIM / (128 * SPLITK);
    constexpr int U = 3;
    static_assert(TOTAL_U * 128 * SPLITK == K_DIM && TOTAL_U % U == 0);
    constexpr int NB = TOTAL_U / U;
    constexpr int CB = K_DIM >> 1;
    constexpr int SB = K_DIM >> 4;
    __shared__ float cs[SPLITK][16][8];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int r0 = blockIdx.x * 16;
    const int ra = r0 + g, rb = r0 + g + 8;
    const bool la = (ra < N), lb = (rb < N);
    const int c_warp = warp * (4 * TOTAL_U) + t;
    const bool lx = (g < B_CONST);
    const __half* xg = lx ? (x_h + (size_t)g * K_DIM) : nullptr;

    float d[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 1
    for (int jb = 0; jb < NB; ++jb) {
        const int c_base = c_warp + jb * (4 * U);
        Nvfp4Chunk32 A[U], B[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int c = c_base + 4 * u;
            if (la) A[u] = load_nvfp4_chunk32(w_data + (size_t)ra * CB, w_scales + (size_t)ra * SB, c);
            else { A[u].q = make_uint4(0, 0, 0, 0); A[u].s = 0; }
            if (lb) B[u] = load_nvfp4_chunk32(w_data + (size_t)rb * CB, w_scales + (size_t)rb * SB, c);
            else { B[u].q = make_uint4(0, 0, 0, 0); B[u].s = 0; }
        }
        if (jb == 0) {
            pdl_wait_fenced(zero);
            pdl_trigger();
        }

        uint32_t xw[U][16];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int col = (c_base + 4 * u) * 32;
            if (lx) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    uint4 q = ld_dep_u4(xg + col + 8 * v);
                    xw[u][4 * v + 0] = q.x; xw[u][4 * v + 1] = q.y;
                    xw[u][4 * v + 2] = q.z; xw[u][4 * v + 3] = q.w;
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) xw[u][v] = 0;
            }
        }

        uint32_t dep = 0;
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            dep ^= A[u].q.x ^ A[u].q.w ^ B[u].q.x ^ B[u].q.w ^ xw[u][0];
        }
        dep &= zero;

        #pragma unroll
        for (int u = 0; u < U; ++u) {
            uint32_t a[16], b[16];
            decode_nvfp4_chunk32_f16(A[u], a);
            decode_nvfp4_chunk32_f16(B[u], b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                     a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                     xw[u][2 * i] ^ dep, xw[u][2 * i + 1],
                                     d[0], d[1], d[2], d[3]);
            }
        }
    }

    cs[warp][g][2 * t]     = d[0];
    cs[warp][g][2 * t + 1] = d[1];
    cs[warp][g + 8][2 * t]     = d[2];
    cs[warp][g + 8][2 * t + 1] = d[3];
    __syncthreads();

    for (int k = threadIdx.x; k < 16 * B_CONST; k += SPLITK * 32) {
        int r = k / B_CONST;
        int m = k % B_CONST;
        int row = r0 + r;
        if (row < N) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < SPLITK; ++w) s += cs[w][r][m];
            s *= global_scale;
            if constexpr (ADD_RES) y[(size_t)m * N + row] += s;
            else                   y[(size_t)m * N + row] = s;
        }
    }
}

// FUSED Q+R+Gate+Sink (6552) + K (1024) + V (1024) = 8600 rows in ONE launch!
// B=1: 1075 CTAs x 128 threads (16 threads/row), pre-issues ALL 9 weight chunks before pdl_wait_fenced!
__global__ void __launch_bounds__(128)
qkvrg_fused_nvfp4_b1_pdl_kernel(
    const uint8_t* __restrict__ wq_d, const uint8_t* __restrict__ wq_s, float q_gs,
    const uint8_t* __restrict__ wk_d, const uint8_t* __restrict__ wk_s, float k_gs,
    const uint8_t* __restrict__ wv_d, const uint8_t* __restrict__ wv_s, float v_gs,
    const __half* __restrict__ xq_h,
    const __half* __restrict__ xk_h,
    const __half* __restrict__ xv_h,
    float* __restrict__ y_qrgs,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    uint32_t zero) {
    constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552
    constexpr int N_K = KV_DIM;                            // 1024
    constexpr int N_TOT = N_Q + N_K + N_K;                 // 8600

    int hw = threadIdx.x >> 4;
    int lane = threadIdx.x & 15;
    int g_row = blockIdx.x * 8 + hw;
    if (g_row >= N_TOT) return;

    const uint8_t* w_data;
    const uint8_t* w_scales;
    float gscale;
    const __half* x_h;
    float* y_out;
    int row;
    if (g_row < N_Q) {
        w_data = wq_d; w_scales = wq_s; gscale = q_gs;
        x_h = xq_h; y_out = y_qrgs; row = g_row;
    } else if (g_row < N_Q + N_K) {
        w_data = wk_d; w_scales = wk_s; gscale = k_gs;
        x_h = xk_h; y_out = y_k; row = g_row - N_Q;
    } else {
        w_data = wv_d; w_scales = wv_s; gscale = v_gs;
        x_h = xv_h; y_out = y_v; row = g_row - (N_Q + N_K);
    }

    constexpr int NBLK = HIDDEN >> 4; // 144 = 9 * 16
    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (HIDDEN >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * NBLK;

    uint2 w8[9];
    uint8_t s8[9];
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int blk = lane + iter * 16;
        w8[iter] = ld_stream_u2(w_row + blk);
        s8[iter] = ld_scale_u8(s_row + blk);
    }
    pdl_wait_fenced(zero);
    pdl_trigger();

    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int koff = (lane + iter * 16) << 4;
        uint4 xv0 = ld_dep_u4(x_h + koff);
        uint4 xv1 = ld_dep_u4(x_h + koff + 8);
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8[iter].x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8[iter].y, wh[4], wh[5], wh[6], wh[7]);
        float sc = cvt_e4m3_to_float(s8[iter]);
        const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
        __half2 p0 = __hmul2(wh[0], xb0[0]);
        __half2 p1 = __hmul2(wh[1], xb0[1]);
        p0 = __hfma2(wh[2], xb0[2], p0);
        p1 = __hfma2(wh[3], xb0[3], p1);
        p0 = __hfma2(wh[4], xb1[0], p0);
        p1 = __hfma2(wh[5], xb1[1], p1);
        p0 = __hfma2(wh[6], xb1[2], p0);
        p1 = __hfma2(wh[7], xb1[3], p1);
        float2 pf = __half22float2(__hadd2(p0, p1));
        acc = fmaf(pf.x + pf.y, sc, acc);
    }
    float s = half_warp_sum(acc) * gscale;
    if (lane == 0) y_out[row] = s;
}

// B=2..8 Tensor Core FUSED QKV+R+G+S: 538 non-straddling CTAs (410 Q + 64 K + 64 V), 6 warps (192 threads)!
template <int B_CONST>
__global__ void __launch_bounds__(192)
qkvrg_fused_nvfp4_mma_kernel(
    const uint8_t* __restrict__ wq_d, const uint8_t* __restrict__ wq_s, float q_gs,
    const uint8_t* __restrict__ wk_d, const uint8_t* __restrict__ wk_s, float k_gs,
    const uint8_t* __restrict__ wv_d, const uint8_t* __restrict__ wv_s, float v_gs,
    const __half* __restrict__ xq_h,
    const __half* __restrict__ xk_h,
    const __half* __restrict__ xv_h,
    float* __restrict__ y_qrgs,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    uint32_t zero) {
    constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552 -> 410 tiles of 16
    constexpr int T_Q = (N_Q + 15) / 16;                   // 410
    constexpr int T_K = KV_DIM / 16;                       // 64
    constexpr int SPLITK = 6;
    constexpr int U = 3;                                   // 6 * 3 * 128 = 2304 EXACT!
    constexpr int CB = HIDDEN >> 1;
    constexpr int SB = HIDDEN >> 4;
    __shared__ float cs[SPLITK][16][8];

    int blk = blockIdx.x;
    const uint8_t* w_data;
    const uint8_t* w_scales;
    float gscale;
    const __half* x_h;
    float* y_out;
    int r0, N_out;
    if (blk < T_Q) {
        w_data = wq_d; w_scales = wq_s; gscale = q_gs;
        x_h = xq_h; y_out = y_qrgs; r0 = blk * 16; N_out = N_Q;
    } else if (blk < T_Q + T_K) {
        w_data = wk_d; w_scales = wk_s; gscale = k_gs;
        x_h = xk_h; y_out = y_k; r0 = (blk - T_Q) * 16; N_out = KV_DIM;
    } else {
        w_data = wv_d; w_scales = wv_s; gscale = v_gs;
        x_h = xv_h; y_out = y_v; r0 = (blk - T_Q - T_K) * 16; N_out = KV_DIM;
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int ra = r0 + g, rb = r0 + g + 8;
    const bool la = (ra < N_out), lb = (rb < N_out);
    const int c_base = warp * (4 * U) + t;

    Nvfp4Chunk32 A[U], B[U];
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        int c = c_base + 4 * u;
        if (la) A[u] = load_nvfp4_chunk32(w_data + (size_t)ra * CB, w_scales + (size_t)ra * SB, c);
        else { A[u].q = make_uint4(0, 0, 0, 0); A[u].s = 0; }
        if (lb) B[u] = load_nvfp4_chunk32(w_data + (size_t)rb * CB, w_scales + (size_t)rb * SB, c);
        else { B[u].q = make_uint4(0, 0, 0, 0); B[u].s = 0; }
    }
    pdl_wait_fenced(zero);
    pdl_trigger();

    uint32_t xw[U][16];
    const bool lx = (g < B_CONST);
    const __half* xg = lx ? (x_h + (size_t)g * HIDDEN) : nullptr;
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        int col = (c_base + 4 * u) * 32;
        if (lx) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint4 q = ld_dep_u4(xg + col + 8 * v);
                xw[u][4 * v + 0] = q.x; xw[u][4 * v + 1] = q.y;
                xw[u][4 * v + 2] = q.z; xw[u][4 * v + 3] = q.w;
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) xw[u][v] = 0;
        }
    }

    uint32_t dep = 0;
    #pragma unroll
    for (int u = 0; u < U; ++u) dep ^= A[u].q.x ^ A[u].q.w ^ B[u].q.x ^ B[u].q.w ^ xw[u][0];
    dep &= zero;

    float d[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        uint32_t a[16], b[16];
        decode_nvfp4_chunk32_f16(A[u], a);
        decode_nvfp4_chunk32_f16(B[u], b);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                 a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                 xw[u][2 * i] ^ dep, xw[u][2 * i + 1],
                                 d[0], d[1], d[2], d[3]);
        }
    }

    cs[warp][g][2 * t]     = d[0];
    cs[warp][g][2 * t + 1] = d[1];
    cs[warp][g + 8][2 * t]     = d[2];
    cs[warp][g + 8][2 * t + 1] = d[3];
    __syncthreads();

    for (int k = threadIdx.x; k < 16 * B_CONST; k += SPLITK * 32) {
        int r = k / B_CONST;
        int m = k % B_CONST;
        int row = r0 + r;
        if (row < N_out) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < SPLITK; ++w) s += cs[w][r][m];
            y_out[(size_t)m * N_out + row] = s * gscale;
        }
    }
}

// ============================================================================
// Blackwell Tensor Core (mma.m16n8k16.f32.f16) + PDL Dense FP8 GEMVs (B = 1..8)
// Every E4M3 weight converts losslessly to FP16 and accumulates in exact FP32!
// ============================================================================

template <bool ADD_RES, int ITERS>
__global__ void __launch_bounds__(128)
linear_fp8_b1_pdl_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const __half* __restrict__ x_h,
    float* __restrict__ y,
    int N, int K, uint32_t zero) {
    int hw = threadIdx.x >> 4;
    int lane = threadIdx.x & 15;
    int row = blockIdx.x * 8 + hw;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    float rscale = row_scales[row];

    // Prefetch first 3 weight chunks before PDL wait to overlap predecessor tail
    uint4 w_pre[3];
    #pragma unroll
    for (int p = 0; p < 3; ++p) w_pre[p] = ld_stream_u4(w_row + lane + p * 16);
    pdl_wait_fenced(zero);
    pdl_trigger();

    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < ITERS; ++iter) {
        uint4 w16 = (iter < 3) ? w_pre[iter] : ld_stream_u4(w_row + lane + iter * 16);
        int koff = (lane + iter * 16) << 4;
        uint4 xv0 = ld_dep_u4(x_h + koff);
        uint4 xv1 = ld_dep_u4(x_h + koff + 8);
        const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
        const uint32_t ws[4] = {w16.x, w16.y, w16.z, w16.w};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float2 wf0 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[j] & 0xFFFFu)));
            float2 wf1 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[j] >> 16)));
            float2 xf0 = __half22float2(xb0[2 * j + 0]);
            float2 xf1 = __half22float2(xb0[2 * j + 1]);
            acc = fmaf(wf0.x, xf0.x, acc); acc = fmaf(wf0.y, xf0.y, acc);
            acc = fmaf(wf1.x, xf1.x, acc); acc = fmaf(wf1.y, xf1.y, acc);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float2 wf0 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[2 + j] & 0xFFFFu)));
            float2 wf1 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[2 + j] >> 16)));
            float2 xf0 = __half22float2(xb1[2 * j + 0]);
            float2 xf1 = __half22float2(xb1[2 * j + 1]);
            acc = fmaf(wf0.x, xf0.x, acc); acc = fmaf(wf0.y, xf0.y, acc);
            acc = fmaf(wf1.x, xf1.x, acc); acc = fmaf(wf1.y, xf1.y, acc);
        }
    }
    float s = half_warp_sum(acc) * rscale;
    if (lane == 0) {
        if constexpr (ADD_RES) y[row] += s;
        else                   y[row] = s;
    }
}

__global__ void __launch_bounds__(128)
qkvrg_fused_fp8_b1_pdl_kernel(
    const uint8_t* __restrict__ wq_d, const float* __restrict__ wq_rs,
    const uint8_t* __restrict__ wk_d, const float* __restrict__ wk_rs,
    const uint8_t* __restrict__ wv_d, const float* __restrict__ wv_rs,
    const __half* __restrict__ xq_h,
    const __half* __restrict__ xk_h,
    const __half* __restrict__ xv_h,
    float* __restrict__ y_qrgs,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    uint32_t zero) {
    constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552
    constexpr int N_K = KV_DIM;                            // 1024
    constexpr int N_TOT = N_Q + 2 * N_K;                   // 8600

    int hw = threadIdx.x >> 4;
    int lane = threadIdx.x & 15;
    int g_row = blockIdx.x * 8 + hw;
    if (g_row >= N_TOT) return;

    const uint8_t* w_data;
    const float* rscales;
    const __half* x_h;
    float* y_out;
    int row;
    if (g_row < N_Q) {
        w_data = wq_d; rscales = wq_rs; x_h = xq_h; y_out = y_qrgs; row = g_row;
    } else if (g_row < N_Q + N_K) {
        w_data = wk_d; rscales = wk_rs; x_h = xk_h; y_out = y_k; row = g_row - N_Q;
    } else {
        w_data = wv_d; rscales = wv_rs; x_h = xv_h; y_out = y_v; row = g_row - (N_Q + N_K);
    }

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * HIDDEN);
    float rscale = rscales[row];

    uint4 w_pre[3];
    #pragma unroll
    for (int p = 0; p < 3; ++p) w_pre[p] = ld_stream_u4(w_row + lane + p * 16);
    pdl_wait_fenced(zero);
    pdl_trigger();

    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        uint4 w16 = (iter < 3) ? w_pre[iter] : ld_stream_u4(w_row + lane + iter * 16);
        int koff = (lane + iter * 16) << 4;
        uint4 xv0 = ld_dep_u4(x_h + koff);
        uint4 xv1 = ld_dep_u4(x_h + koff + 8);
        const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
        const uint32_t ws[4] = {w16.x, w16.y, w16.z, w16.w};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float2 wf0 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[j] & 0xFFFFu)));
            float2 wf1 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[j] >> 16)));
            float2 xf0 = __half22float2(xb0[2 * j + 0]);
            float2 xf1 = __half22float2(xb0[2 * j + 1]);
            acc = fmaf(wf0.x, xf0.x, acc); acc = fmaf(wf0.y, xf0.y, acc);
            acc = fmaf(wf1.x, xf1.x, acc); acc = fmaf(wf1.y, xf1.y, acc);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float2 wf0 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[2 + j] & 0xFFFFu)));
            float2 wf1 = __half22float2(cvt_e4m3x2_to_half2(uint16_t(ws[2 + j] >> 16)));
            float2 xf0 = __half22float2(xb1[2 * j + 0]);
            float2 xf1 = __half22float2(xb1[2 * j + 1]);
            acc = fmaf(wf0.x, xf0.x, acc); acc = fmaf(wf0.y, xf0.y, acc);
            acc = fmaf(wf1.x, xf1.x, acc); acc = fmaf(wf1.y, xf1.y, acc);
        }
    }
    float s = half_warp_sum(acc) * rscale;
    if (lane == 0) y_out[row] = s;
}

template <int B_CONST, bool ADD_RES, int K_DIM, int SPLITK>
__global__ void __launch_bounds__(SPLITK * 32)
linear_fp8_mma_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const __half* __restrict__ x_h,
    float* __restrict__ y,
    int N, uint32_t zero) {
    constexpr int TOTAL_U = K_DIM / (128 * SPLITK);
    static_assert(TOTAL_U * 128 * SPLITK == K_DIM);
    __shared__ float cs[SPLITK][16][8];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int r0 = blockIdx.x * 16;
    const int ra = r0 + g, rb = r0 + g + 8;
    const bool la = (ra < N), lb = (rb < N);
    const int c_warp = warp * (4 * TOTAL_U) + t;
    const bool lx = (g < B_CONST);
    const __half* xg = lx ? (x_h + (size_t)g * K_DIM) : nullptr;

    Fp8Chunk32 A_cur, B_cur;
    if (la) A_cur = load_fp8_chunk32(w_data + (size_t)ra * K_DIM, c_warp);
    else { A_cur.q0 = make_uint4(0, 0, 0, 0); A_cur.q1 = make_uint4(0, 0, 0, 0); }
    if (lb) B_cur = load_fp8_chunk32(w_data + (size_t)rb * K_DIM, c_warp);
    else { B_cur.q0 = make_uint4(0, 0, 0, 0); B_cur.q1 = make_uint4(0, 0, 0, 0); }

    pdl_wait_fenced(zero);
    pdl_trigger();

    float d[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int u = 0; u < TOTAL_U; ++u) {
        Fp8Chunk32 A_nxt, B_nxt;
        if (u + 1 < TOTAL_U) {
            int c_nxt = c_warp + 4 * (u + 1);
            if (la) A_nxt = load_fp8_chunk32(w_data + (size_t)ra * K_DIM, c_nxt);
            else { A_nxt.q0 = make_uint4(0, 0, 0, 0); A_nxt.q1 = make_uint4(0, 0, 0, 0); }
            if (lb) B_nxt = load_fp8_chunk32(w_data + (size_t)rb * K_DIM, c_nxt);
            else { B_nxt.q0 = make_uint4(0, 0, 0, 0); B_nxt.q1 = make_uint4(0, 0, 0, 0); }
        }
        uint32_t xw[16];
        int col = (c_warp + 4 * u) * 32;
        if (lx) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint4 q = ld_dep_u4(xg + col + 8 * v);
                xw[4 * v + 0] = q.x; xw[4 * v + 1] = q.y;
                xw[4 * v + 2] = q.z; xw[4 * v + 3] = q.w;
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) xw[v] = 0;
        }
        uint32_t dep = (A_cur.q0.x ^ A_cur.q1.w ^ B_cur.q0.x ^ B_cur.q1.w ^ xw[0]) & zero;
        uint32_t a[16], b[16];
        decode_fp8_chunk32_f16(A_cur, a);
        decode_fp8_chunk32_f16(B_cur, b);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                 a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                 xw[2 * i] ^ dep, xw[2 * i + 1],
                                 d[0], d[1], d[2], d[3]);
        }
        if (u + 1 < TOTAL_U) { A_cur = A_nxt; B_cur = B_nxt; }
    }

    cs[warp][g][2 * t]     = d[0];
    cs[warp][g][2 * t + 1] = d[1];
    cs[warp][g + 8][2 * t]     = d[2];
    cs[warp][g + 8][2 * t + 1] = d[3];
    __syncthreads();

    for (int k = threadIdx.x; k < 16 * B_CONST; k += SPLITK * 32) {
        int r = k / B_CONST;
        int m = k % B_CONST;
        int row = r0 + r;
        if (row < N) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < SPLITK; ++w) s += cs[w][r][m];
            s *= row_scales[row];
            if constexpr (ADD_RES) y[(size_t)m * N + row] += s;
            else                   y[(size_t)m * N + row] = s;
        }
    }
}

template <int B_CONST>
__global__ void __launch_bounds__(192)
qkvrg_fused_fp8_mma_kernel(
    const uint8_t* __restrict__ wq_d, const float* __restrict__ wq_rs,
    const uint8_t* __restrict__ wk_d, const float* __restrict__ wk_rs,
    const uint8_t* __restrict__ wv_d, const float* __restrict__ wv_rs,
    const __half* __restrict__ xq_h,
    const __half* __restrict__ xk_h,
    const __half* __restrict__ xv_h,
    float* __restrict__ y_qrgs,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    uint32_t zero) {
    constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552 -> 410 tiles of 16
    constexpr int T_Q = (N_Q + 15) / 16;                   // 410
    constexpr int T_K = KV_DIM / 16;                       // 64
    constexpr int SPLITK = 6;
    constexpr int TOTAL_U = 3;                             // 6 * 3 * 128 = 2304 EXACT!
    __shared__ float cs[SPLITK][16][8];

    int blk = blockIdx.x;
    const uint8_t* w_data;
    const float* rscales;
    const __half* x_h;
    float* y_out;
    int r0, N_out;
    if (blk < T_Q) {
        w_data = wq_d; rscales = wq_rs; x_h = xq_h; y_out = y_qrgs; r0 = blk * 16; N_out = N_Q;
    } else if (blk < T_Q + T_K) {
        w_data = wk_d; rscales = wk_rs; x_h = xk_h; y_out = y_k; r0 = (blk - T_Q) * 16; N_out = KV_DIM;
    } else {
        w_data = wv_d; rscales = wv_rs; x_h = xv_h; y_out = y_v; r0 = (blk - T_Q - T_K) * 16; N_out = KV_DIM;
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int ra = r0 + g, rb = r0 + g + 8;
    const bool la = (ra < N_out), lb = (rb < N_out);
    const int c_base = warp * (4 * TOTAL_U) + t;
    const bool lx = (g < B_CONST);
    const __half* xg = lx ? (x_h + (size_t)g * HIDDEN) : nullptr;

    Fp8Chunk32 A_cur, B_cur;
    if (la) A_cur = load_fp8_chunk32(w_data + (size_t)ra * HIDDEN, c_base);
    else { A_cur.q0 = make_uint4(0, 0, 0, 0); A_cur.q1 = make_uint4(0, 0, 0, 0); }
    if (lb) B_cur = load_fp8_chunk32(w_data + (size_t)rb * HIDDEN, c_base);
    else { B_cur.q0 = make_uint4(0, 0, 0, 0); B_cur.q1 = make_uint4(0, 0, 0, 0); }

    pdl_wait_fenced(zero);
    pdl_trigger();

    float d[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int u = 0; u < TOTAL_U; ++u) {
        Fp8Chunk32 A_nxt, B_nxt;
        if (u + 1 < TOTAL_U) {
            int c_nxt = c_base + 4 * (u + 1);
            if (la) A_nxt = load_fp8_chunk32(w_data + (size_t)ra * HIDDEN, c_nxt);
            else { A_nxt.q0 = make_uint4(0, 0, 0, 0); A_nxt.q1 = make_uint4(0, 0, 0, 0); }
            if (lb) B_nxt = load_fp8_chunk32(w_data + (size_t)rb * HIDDEN, c_nxt);
            else { B_nxt.q0 = make_uint4(0, 0, 0, 0); B_nxt.q1 = make_uint4(0, 0, 0, 0); }
        }
        uint32_t xw[16];
        int col = (c_base + 4 * u) * 32;
        if (lx) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint4 q = ld_dep_u4(xg + col + 8 * v);
                xw[4 * v + 0] = q.x; xw[4 * v + 1] = q.y;
                xw[4 * v + 2] = q.z; xw[4 * v + 3] = q.w;
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) xw[v] = 0;
        }
        uint32_t dep = (A_cur.q0.x ^ A_cur.q1.w ^ B_cur.q0.x ^ B_cur.q1.w ^ xw[0]) & zero;
        uint32_t a[16], b[16];
        decode_fp8_chunk32_f16(A_cur, a);
        decode_fp8_chunk32_f16(B_cur, b);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                 a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                 xw[2 * i] ^ dep, xw[2 * i + 1],
                                 d[0], d[1], d[2], d[3]);
        }
        if (u + 1 < TOTAL_U) { A_cur = A_nxt; B_cur = B_nxt; }
    }

    cs[warp][g][2 * t]     = d[0];
    cs[warp][g][2 * t + 1] = d[1];
    cs[warp][g + 8][2 * t]     = d[2];
    cs[warp][g + 8][2 * t + 1] = d[3];
    __syncthreads();

    for (int k = threadIdx.x; k < 16 * B_CONST; k += SPLITK * 32) {
        int r = k / B_CONST;
        int m = k % B_CONST;
        int row = r0 + r;
        if (row < N_out) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < SPLITK; ++w) s += cs[w][r][m];
            y_out[(size_t)m * N_out + row] = s * rscales[row];
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_fp8_h_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const __half* __restrict__ x_h,
    float* __restrict__ y,
    int N, int K) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    int n_vec16 = K >> 4;
    float rscale = row_scales[row];

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < n_vec16; v += 32) {
        uint4 w16 = __ldg(w_row + v);
        __half2 wh[8];
        wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
        wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
        wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
        wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
        wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
        wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
        wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
        wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));

        int koff = v << 4;
        float2 wf[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) wf[j] = __half22float2(wh[j]);
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * K + koff);
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 xf = __half22float2(xb0[j]);
                acc[b] = fmaf(wf[j].x, xf.x, acc[b]);
                acc[b] = fmaf(wf[j].y, xf.y, acc[b]);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 xf = __half22float2(xb1[j]);
                acc[b] = fmaf(wf[4 + j].x, xf.x, acc[b]);
                acc[b] = fmaf(wf[4 + j].y, xf.y, acc[b]);
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]) * rscale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_bf16_nosmem_t_kernel(
    const __nv_bfloat16* __restrict__ w_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    int n_vec8 = K >> 3;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < n_vec8; v += 32) {
        uint4 w8 = __ldg(w_row + v);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
        float2 wf0 = __bfloat1622float2(wb[0]);
        float2 wf1 = __bfloat1622float2(wb[1]);
        float2 wf2 = __bfloat1622float2(wb[2]);
        float2 wf3 = __bfloat1622float2(wb[3]);
        int koff = v << 3;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const float4* xf = reinterpret_cast<const float4*>(x + (size_t)b * K + koff);
            float4 x0 = __ldg(xf), x1 = __ldg(xf + 1);
            acc[b] = fmaf(wf0.x, x0.x, acc[b]); acc[b] = fmaf(wf0.y, x0.y, acc[b]);
            acc[b] = fmaf(wf1.x, x0.z, acc[b]); acc[b] = fmaf(wf1.y, x0.w, acc[b]);
            acc[b] = fmaf(wf2.x, x1.x, acc[b]); acc[b] = fmaf(wf2.y, x1.y, acc[b]);
            acc[b] = fmaf(wf3.x, x1.z, acc[b]); acc[b] = fmaf(wf3.y, x1.w, acc[b]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]);
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_nvfp4_f32_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K) {
    // Zero-SMEM: converts x on the fly from L1 (used for DSpark/general fallback)
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (K >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * (K >> 4);
    int n_blocks16 = K >> 4;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int blk = lane; blk < n_blocks16; blk += 32) {
        uint2 w8 = __ldg(w_row + blk);
        float sc = cvt_e4m3_to_float(__ldg(s_row + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const float4* xf = reinterpret_cast<const float4*>(x + (size_t)b * K + koff);
            float4 f0 = __ldg(xf), f1 = __ldg(xf + 1), f2 = __ldg(xf + 2), f3 = __ldg(xf + 3);
            __half2 p0 = __hmul2(wh[0], __floats2half2_rn(f0.x, f0.y));
            __half2 p1 = __hmul2(wh[1], __floats2half2_rn(f0.z, f0.w));
            p0 = __hfma2(wh[2], __floats2half2_rn(f1.x, f1.y), p0);
            p1 = __hfma2(wh[3], __floats2half2_rn(f1.z, f1.w), p1);
            p0 = __hfma2(wh[4], __floats2half2_rn(f2.x, f2.y), p0);
            p1 = __hfma2(wh[5], __floats2half2_rn(f2.z, f2.w), p1);
            p0 = __hfma2(wh[6], __floats2half2_rn(f3.x, f3.y), p0);
            p1 = __hfma2(wh[7], __floats2half2_rn(f3.z, f3.w), p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]) * global_scale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_fp8_f32_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    int n_vec16 = K >> 4;
    float rscale = row_scales[row];

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < n_vec16; v += 32) {
        uint4 w16 = __ldg(w_row + v);
        __half2 wh[8];
        wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
        wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
        wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
        wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
        wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
        wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
        wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
        wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));
        int koff = v << 4;
        float2 wf[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) wf[j] = __half22float2(wh[j]);
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const float4* xf = reinterpret_cast<const float4*>(x + (size_t)b * K + koff);
            float4 f0 = __ldg(xf), f1 = __ldg(xf + 1), f2 = __ldg(xf + 2), f3 = __ldg(xf + 3);
            acc[b] = fmaf(wf[0].x, f0.x, acc[b]); acc[b] = fmaf(wf[0].y, f0.y, acc[b]);
            acc[b] = fmaf(wf[1].x, f0.z, acc[b]); acc[b] = fmaf(wf[1].y, f0.w, acc[b]);
            acc[b] = fmaf(wf[2].x, f1.x, acc[b]); acc[b] = fmaf(wf[2].y, f1.y, acc[b]);
            acc[b] = fmaf(wf[3].x, f1.z, acc[b]); acc[b] = fmaf(wf[3].y, f1.w, acc[b]);
            acc[b] = fmaf(wf[4].x, f2.x, acc[b]); acc[b] = fmaf(wf[4].y, f2.y, acc[b]);
            acc[b] = fmaf(wf[5].x, f2.z, acc[b]); acc[b] = fmaf(wf[5].y, f2.w, acc[b]);
            acc[b] = fmaf(wf[6].x, f3.x, acc[b]); acc[b] = fmaf(wf[6].y, f3.y, acc[b]);
            acc[b] = fmaf(wf[7].x, f3.z, acc[b]); acc[b] = fmaf(wf[7].y, f3.w, acc[b]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]) * rscale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST>
static void dispatch_linear_b(
    const LinearWeight& w, const float* x, float* y, bool add_res, cudaStream_t stream) {
    int grid = (w.rows + 3) / 4;
    if (w.kind == WeightKind::NVFP4) {
        if (add_res) linear_nvfp4_f32_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x, y, w.rows, w.cols);
        else linear_nvfp4_f32_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x, y, w.rows, w.cols);
    } else if (w.kind == WeightKind::FP8) {
        if (add_res) linear_fp8_f32_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x, y, w.rows, w.cols);
        else linear_fp8_f32_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x, y, w.rows, w.cols);
    } else {
        if (add_res) linear_bf16_nosmem_t_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), x, y, w.rows, w.cols);
        else linear_bf16_nosmem_t_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), x, y, w.rows, w.cols);
    }
}

template <int B_CONST>
static void dispatch_linear_h_b(
    const LinearWeight& w, const __half* x_h, float* y, bool add_res, cudaStream_t stream) {
    if (w.kind == WeightKind::NVFP4) {
        if constexpr (B_CONST == 1) {
            if (w.cols == 3072) {
                int grid = (w.rows + 7) / 8;
                if (add_res) launch_pdl(linear_nvfp4_b1_pdl_kernel<true, 12>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols, 0u);
                else launch_pdl(linear_nvfp4_b1_pdl_kernel<false, 12>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols, 0u);
                return;
            } else if (w.cols == 2304) {
                int grid = (w.rows + 7) / 8;
                if (add_res) launch_pdl(linear_nvfp4_b1_pdl_kernel<true, 9>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols, 0u);
                else launch_pdl(linear_nvfp4_b1_pdl_kernel<false, 9>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols, 0u);
                return;
            }
        } else {
            if (w.cols == 3072) {
                int grid = (w.rows + 15) / 16;
                if (add_res) launch_pdl(linear_nvfp4_mma_kernel<B_CONST, true, 3072, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, 0u);
                else launch_pdl(linear_nvfp4_mma_kernel<B_CONST, false, 3072, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, 0u);
                return;
            } else if (w.cols == 2304) {
                int grid = (w.rows + 15) / 16;
                if (add_res) launch_pdl(linear_nvfp4_mma_kernel<B_CONST, true, 2304, 6>, dim3(grid), 192, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, 0u);
                else launch_pdl(linear_nvfp4_mma_kernel<B_CONST, false, 2304, 6>, dim3(grid), 192, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, 0u);
                return;
            } else if (w.cols == 9216) {
                int grid = (w.rows + 15) / 16;
                if (add_res) launch_pdl(linear_nvfp4_mma_kernel<B_CONST, true, 9216, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, 0u);
                else launch_pdl(linear_nvfp4_mma_kernel<B_CONST, false, 9216, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, 0u);
                return;
            }
        }
        bool use_half_warp = (((w.cols >> 4) & 31) != 0);
        if (use_half_warp) {
            int grid = (w.rows + 7) / 8;
            if (add_res) linear_nvfp4_h_kernel<B_CONST, true, 16><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
            else linear_nvfp4_h_kernel<B_CONST, false, 16><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
        } else {
            int grid = (w.rows + 3) / 4;
            if (add_res) linear_nvfp4_h_kernel<B_CONST, true, 32><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
            else linear_nvfp4_h_kernel<B_CONST, false, 32><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
        }
    } else if (w.kind == WeightKind::FP8) {
        if constexpr (B_CONST == 1) {
            if (w.cols == 3072) {
                int grid = (w.rows + 7) / 8;
                if (add_res) launch_pdl(linear_fp8_b1_pdl_kernel<true, 12>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols, 0u);
                else launch_pdl(linear_fp8_b1_pdl_kernel<false, 12>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols, 0u);
                return;
            } else if (w.cols == 2304) {
                int grid = (w.rows + 7) / 8;
                if (add_res) launch_pdl(linear_fp8_b1_pdl_kernel<true, 9>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols, 0u);
                else launch_pdl(linear_fp8_b1_pdl_kernel<false, 9>, dim3(grid), 128, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols, 0u);
                return;
            }
        } else {
            if (w.cols == 3072) {
                int grid = (w.rows + 15) / 16;
                if (add_res) launch_pdl(linear_fp8_mma_kernel<B_CONST, true, 3072, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, 0u);
                else launch_pdl(linear_fp8_mma_kernel<B_CONST, false, 3072, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, 0u);
                return;
            } else if (w.cols == 2304) {
                int grid = (w.rows + 15) / 16;
                if (add_res) launch_pdl(linear_fp8_mma_kernel<B_CONST, true, 2304, 6>, dim3(grid), 192, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, 0u);
                else launch_pdl(linear_fp8_mma_kernel<B_CONST, false, 2304, 6>, dim3(grid), 192, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, 0u);
                return;
            } else if (w.cols == 9216) {
                int grid = (w.rows + 15) / 16;
                if (add_res) launch_pdl(linear_fp8_mma_kernel<B_CONST, true, 9216, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, 0u);
                else launch_pdl(linear_fp8_mma_kernel<B_CONST, false, 9216, 8>, dim3(grid), 256, stream,
                    static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, 0u);
                return;
            }
        }
        int grid = (w.rows + 3) / 4;
        if (add_res) linear_fp8_h_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols);
        else linear_fp8_h_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols);
    }
}

// ============================================================================
// Fused LM Head + Softcap(30) + GPU Argmax (HALF-WARP NVFP4 -> 0% tail waste!)
// ============================================================================

constexpr int LM_HEAD_BLOCKS_MAX = 1024; // bounded by lm_blk_val / lm_blk_idx [B, 1024]

static int get_lm_head_blocks() {
    static int cached = 0;
    if (cached > 0) return cached;
    int dev = 0, sms = 170;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        int attr = 0;
        if (cudaDeviceGetAttribute(&attr, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && attr > 0) sms = attr;
    }
    // 4 blocks/SM on large GPUs (170 SMs -> 680 exact), 8 blocks/SM on smaller GPUs (26 SMs -> 208)
    int blks = (sms <= 64) ? (sms * 8) : (sms * 4);
    if (blks < 64) blks = 64;
    if (blks > LM_HEAD_BLOCKS_MAX) blks = LM_HEAD_BLOCKS_MAX;
    cached = blks;
    return cached;
}

template <int B_CONST, int KIND>
__global__ void __launch_bounds__(256)
lm_head_stage1_kernel(
    const void* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ row_scales,
    float global_scale,
    const float* __restrict__ x,
    float* __restrict__ out_logits,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx,
    const SampleConfig* __restrict__ sample_cfg,
    int V) {
    const bool write_logits = (out_logits != nullptr) && (sample_cfg == nullptr || sample_cfg->inv_temp != 0.0f);
    __shared__ uint16_t s_x16[B_CONST * HIDDEN];
    int tid = threadIdx.x;
    int total_x = B_CONST * HIDDEN;
    pdl_wait();
    pdl_trigger();
    if constexpr (KIND == 0) {
        for (int i = tid; i < total_x; i += 256) {
            __nv_bfloat16 bv = __float2bfloat16(ld_dep_f32(x + i));
            s_x16[i] = *reinterpret_cast<uint16_t*>(&bv);
        }
    } else {
        for (int i = tid; i < total_x; i += 256) {
            __half hv = __float2half(ld_dep_f32(x + i));
            s_x16[i] = *reinterpret_cast<uint16_t*>(&hv);
        }
    }
    __syncthreads();
    const __half* s_xh = reinterpret_cast<const __half*>(s_x16);
    const __nv_bfloat16* s_xbf = reinterpret_cast<const __nv_bfloat16*>(s_x16);

    constexpr int TPR = (KIND == 2) ? 16 : 32;
    constexpr int RPB = 256 / TPR; // 16 for NVFP4, 8 for FP8/BF16
    int r_in_blk = (TPR == 16) ? (tid >> 4) : (tid >> 5);
    int lane = tid & (TPR - 1);
    int global_r = blockIdx.x * RPB + r_in_blk;
    int total_r = gridDim.x * RPB;

    float best_val[B_CONST];
    int32_t best_idx[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { best_val[b] = -1e30f; best_idx[b] = 0; }

    for (int row = global_r; row < V; row += total_r) {
        float acc[B_CONST];
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

        if constexpr (KIND == 2) {
            const uint2* wr = reinterpret_cast<const uint2*>(
                static_cast<const uint8_t*>(w_data) + (size_t)row * (HIDDEN >> 1));
            const uint8_t* sr = w_scales + (size_t)row * (HIDDEN >> 4);
            #pragma unroll
            for (int iter = 0; iter < 9; ++iter) {
                int blk = lane + iter * 16;
                uint2 w8 = __ldg(wr + blk);
                float sc = cvt_e4m3_to_float(__ldg(sr + blk));
                __half2 wh[8];
                cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
                cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
                int koff = blk << 4;
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    const __half2* xb = reinterpret_cast<const __half2*>(s_xh + b * HIDDEN + koff);
                    __half2 p0 = __hmul2(wh[0], xb[0]), p1 = __hmul2(wh[1], xb[1]);
                    p0 = __hfma2(wh[2], xb[2], p0); p1 = __hfma2(wh[3], xb[3], p1);
                    p0 = __hfma2(wh[4], xb[4], p0); p1 = __hfma2(wh[5], xb[5], p1);
                    p0 = __hfma2(wh[6], xb[6], p0); p1 = __hfma2(wh[7], xb[7], p1);
                    float2 pf = __half22float2(__hadd2(p0, p1));
                    acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
                }
            }
        } else if constexpr (KIND == 1) {
            const uint4* wr = reinterpret_cast<const uint4*>(
                static_cast<const uint8_t*>(w_data) + (size_t)row * HIDDEN);
            constexpr int NVEC = HIDDEN >> 4;
            for (int v = lane; v < NVEC; v += 32) {
                uint4 w16 = __ldg(wr + v);
                __half2 wh[8];
                wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
                wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
                wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
                wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
                wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
                wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
                wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
                wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));
                int koff = v << 4;
                float2 wf[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) wf[j] = __half22float2(wh[j]);
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    const __half2* xb = reinterpret_cast<const __half2*>(s_xh + b * HIDDEN + koff);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float2 xf = __half22float2(xb[j]);
                        acc[b] = fmaf(wf[j].x, xf.x, acc[b]);
                        acc[b] = fmaf(wf[j].y, xf.y, acc[b]);
                    }
                }
            }
        } else {
            const uint4* wr = reinterpret_cast<const uint4*>(
                static_cast<const __nv_bfloat16*>(w_data) + (size_t)row * HIDDEN);
            constexpr int NVEC = HIDDEN >> 3;
            for (int v = lane; v < NVEC; v += 32) {
                uint4 w8 = __ldg(wr + v);
                const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
                float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
                float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
                int koff = v << 3;
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    uint4 x8 = *reinterpret_cast<const uint4*>(s_xbf + b * HIDDEN + koff);
                    const __nv_bfloat162* xbf = reinterpret_cast<const __nv_bfloat162*>(&x8);
                    float2 xf0 = __bfloat1622float2(xbf[0]), xf1 = __bfloat1622float2(xbf[1]);
                    float2 xf2 = __bfloat1622float2(xbf[2]), xf3 = __bfloat1622float2(xbf[3]);
                    acc[b] = fmaf(wf0.x, xf0.x, acc[b]); acc[b] = fmaf(wf0.y, xf0.y, acc[b]);
                    acc[b] = fmaf(wf1.x, xf1.x, acc[b]); acc[b] = fmaf(wf1.y, xf1.y, acc[b]);
                    acc[b] = fmaf(wf2.x, xf2.x, acc[b]); acc[b] = fmaf(wf2.y, xf2.y, acc[b]);
                    acc[b] = fmaf(wf3.x, xf3.x, acc[b]); acc[b] = fmaf(wf3.y, xf3.y, acc[b]);
                }
            }
        }

        float scale = (KIND == 2) ? global_scale : ((KIND == 1) ? row_scales[row] : 1.0f);
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float val = ((TPR == 16) ? half_warp_sum(acc[b]) : warp_sum(acc[b])) * scale;
            if (lane == 0) {
                if (write_logits) {
                    float cap_val = fast_tanh(val * (1.0f / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                    out_logits[(size_t)b * V + row] = cap_val;
                    if (cap_val > best_val[b]) { best_val[b] = cap_val; best_idx[b] = row; }
                } else {
                    if (val > best_val[b]) { best_val[b] = val; best_idx[b] = row; }
                }
            }
        }
    }

    __shared__ float sm_val[B_CONST][RPB];
    __shared__ int32_t sm_idx[B_CONST][RPB];
    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) { sm_val[b][r_in_blk] = best_val[b]; sm_idx[b][r_in_blk] = best_idx[b]; }
    }
    __syncthreads();
    if (tid < B_CONST) {
        int b = tid;
        float bv = sm_val[b][0];
        int32_t bi = sm_idx[b][0];
        for (int w = 1; w < RPB; ++w) {
            if (sm_val[b][w] > bv || (sm_val[b][w] == bv && sm_idx[b][w] < bi)) { bv = sm_val[b][w]; bi = sm_idx[b][w]; }
        }
        blk_max_val[b * (int)gridDim.x + (int)blockIdx.x] = bv;
        blk_max_idx[b * (int)gridDim.x + (int)blockIdx.x] = bi;
    }
}

// B=2..8 Tensor Core NVFP4 LM Head Stage 1 (680 persistent CTAs, 6 warps = 192 threads, mma.m16n8k16)!
template <int B_CONST>
__global__ void __launch_bounds__(192)
lm_head_nvfp4_mma_stage1_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const float* __restrict__ x,
    float* __restrict__ out_logits,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx,
    const SampleConfig* __restrict__ sample_cfg,
    int V, uint32_t zero) {
    const bool write_logits = (out_logits != nullptr) && (sample_cfg == nullptr || sample_cfg->inv_temp != 0.0f);
    constexpr int SPLITK = 6;
    constexpr int U = 3; // 6 * 3 * 128 = 2304
    constexpr int CB = HIDDEN >> 1;
    constexpr int SB = HIDDEN >> 4;
    __shared__ __half s_xh[B_CONST][HIDDEN];
    __shared__ float cs[SPLITK][16][8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c_base = warp * (4 * U) + t;

    pdl_wait_fenced(zero);
    pdl_trigger();

    int total_x = B_CONST * HIDDEN;
    for (int i = tid; i < total_x; i += 192) {
        s_xh[i / HIDDEN][i % HIDDEN] = __float2half(ld_dep_f32(x + i));
    }
    __syncthreads();

    uint32_t xw[U][16];
    const bool lx = (g < B_CONST);
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        int col = (c_base + 4 * u) * 32;
        if (lx) {
            const uint4* xp = reinterpret_cast<const uint4*>(&s_xh[g][col]);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint4 q = xp[v];
                xw[u][4 * v + 0] = q.x; xw[u][4 * v + 1] = q.y;
                xw[u][4 * v + 2] = q.z; xw[u][4 * v + 3] = q.w;
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) xw[u][v] = 0;
        }
    }

    float best_val[B_CONST];
    int32_t best_idx[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { best_val[b] = -1e30f; best_idx[b] = 0; }

    int n_tiles = (V + 15) / 16;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        int r0 = tile * 16;
        int ra = r0 + g, rb = r0 + g + 8;
        bool la = (ra < V), lb = (rb < V);

        Nvfp4Chunk32 A[U], B[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int c = c_base + 4 * u;
            if (la) A[u] = load_nvfp4_chunk32(w_data + (size_t)ra * CB, w_scales + (size_t)ra * SB, c);
            else { A[u].q = make_uint4(0, 0, 0, 0); A[u].s = 0; }
            if (lb) B[u] = load_nvfp4_chunk32(w_data + (size_t)rb * CB, w_scales + (size_t)rb * SB, c);
            else { B[u].q = make_uint4(0, 0, 0, 0); B[u].s = 0; }
        }
        uint32_t dep = 0;
        #pragma unroll
        for (int u = 0; u < U; ++u) dep ^= A[u].q.x ^ A[u].q.w ^ B[u].q.x ^ B[u].q.w;
        dep &= zero;

        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            uint32_t a[16], b[16];
            decode_nvfp4_chunk32_f16(A[u], a);
            decode_nvfp4_chunk32_f16(B[u], b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                     a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                     xw[u][2 * i] ^ dep, xw[u][2 * i + 1],
                                     d[0], d[1], d[2], d[3]);
            }
        }
        cs[warp][g][2 * t]     = d[0];
        cs[warp][g][2 * t + 1] = d[1];
        cs[warp][g + 8][2 * t]     = d[2];
        cs[warp][g + 8][2 * t + 1] = d[3];
        __syncthreads();

        if (tid < 16) {
            int row = r0 + tid;
            if (row < V) {
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    float s = cs[0][tid][b] + cs[1][tid][b] + cs[2][tid][b]
                            + cs[3][tid][b] + cs[4][tid][b] + cs[5][tid][b];
                    float val = s * global_scale;
                    if (write_logits) {
                        float cap_val = fast_tanh(val * (1.0f / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                        out_logits[(size_t)b * V + row] = cap_val;
                        if (cap_val > best_val[b] || (cap_val == best_val[b] && row < best_idx[b])) {
                            best_val[b] = cap_val; best_idx[b] = row;
                        }
                    } else {
                        if (val > best_val[b] || (val == best_val[b] && row < best_idx[b])) {
                            best_val[b] = val; best_idx[b] = row;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid < 16) {
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float bv = best_val[b];
            int32_t bi = best_idx[b];
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xFFFFu, bv, off);
                int32_t oi = __shfl_xor_sync(0xFFFFu, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (tid == 0) {
                blk_max_val[b * (int)gridDim.x + (int)blockIdx.x] = bv;
                blk_max_idx[b * (int)gridDim.x + (int)blockIdx.x] = bi;
            }
        }
    }
}

// Persistent Blackwell Tensor Core (mma.m16n8k16.f32.f16) + PDL FP8 LM Head for B = 2..8!
template <int B_CONST>
__global__ void __launch_bounds__(192)
lm_head_fp8_mma_stage1_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const float* __restrict__ x,
    float* __restrict__ out_logits,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx,
    const SampleConfig* __restrict__ sample_cfg,
    int V, uint32_t zero) {
    const bool write_logits = (out_logits != nullptr) && (sample_cfg == nullptr || sample_cfg->inv_temp != 0.0f);
    constexpr int SPLITK = 6;
    constexpr int U = 3; // 6 * 3 * 128 = 2304
    __shared__ __half s_xh[B_CONST][HIDDEN];
    __shared__ float cs[SPLITK][16][8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c_base = warp * (4 * U) + t;

    pdl_wait_fenced(zero);
    pdl_trigger();

    int total_x = B_CONST * HIDDEN;
    for (int i = tid; i < total_x; i += 192) {
        s_xh[i / HIDDEN][i % HIDDEN] = __float2half(ld_dep_f32(x + i));
    }
    __syncthreads();

    uint32_t xw[U][16];
    const bool lx = (g < B_CONST);
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        int col = (c_base + 4 * u) * 32;
        if (lx) {
            const uint4* xp = reinterpret_cast<const uint4*>(&s_xh[g][col]);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint4 q = xp[v];
                xw[u][4 * v + 0] = q.x; xw[u][4 * v + 1] = q.y;
                xw[u][4 * v + 2] = q.z; xw[u][4 * v + 3] = q.w;
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) xw[u][v] = 0;
        }
    }

    float best_val[B_CONST];
    int32_t best_idx[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { best_val[b] = -1e30f; best_idx[b] = 0; }

    int n_tiles = (V + 15) / 16;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        int r0 = tile * 16;
        int ra = r0 + g, rb = r0 + g + 8;
        bool la = (ra < V), lb = (rb < V);

        float d[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int c = c_base + 4 * u;
            Fp8Chunk32 Ac, Bc;
            if (la) Ac = load_fp8_chunk32(w_data + (size_t)ra * HIDDEN, c);
            else { Ac.q0 = make_uint4(0, 0, 0, 0); Ac.q1 = make_uint4(0, 0, 0, 0); }
            if (lb) Bc = load_fp8_chunk32(w_data + (size_t)rb * HIDDEN, c);
            else { Bc.q0 = make_uint4(0, 0, 0, 0); Bc.q1 = make_uint4(0, 0, 0, 0); }
            uint32_t dep = (Ac.q0.x ^ Ac.q1.w ^ Bc.q0.x ^ Bc.q1.w) & zero;
            uint32_t a[16], b[16];
            decode_fp8_chunk32_f16(Ac, a);
            decode_fp8_chunk32_f16(Bc, b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                     a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                     xw[u][2 * i] ^ dep, xw[u][2 * i + 1],
                                     d[0], d[1], d[2], d[3]);
            }
        }
        cs[warp][g][2 * t]     = d[0];
        cs[warp][g][2 * t + 1] = d[1];
        cs[warp][g + 8][2 * t]     = d[2];
        cs[warp][g + 8][2 * t + 1] = d[3];
        __syncthreads();

        if (tid < 16) {
            int row = r0 + tid;
            if (row < V) {
                float rscale = row_scales[row];
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    float s = cs[0][tid][b] + cs[1][tid][b] + cs[2][tid][b]
                            + cs[3][tid][b] + cs[4][tid][b] + cs[5][tid][b];
                    float val = s * rscale;
                    if (write_logits) {
                        float cap_val = fast_tanh(val * (1.0f / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                        out_logits[(size_t)b * V + row] = cap_val;
                        if (cap_val > best_val[b] || (cap_val == best_val[b] && row < best_idx[b])) {
                            best_val[b] = cap_val; best_idx[b] = row;
                        }
                    } else {
                        if (val > best_val[b] || (val == best_val[b] && row < best_idx[b])) {
                            best_val[b] = val; best_idx[b] = row;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid < 16) {
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float bv = best_val[b];
            int32_t bi = best_idx[b];
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xFFFFu, bv, off);
                int32_t oi = __shfl_xor_sync(0xFFFFu, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (tid == 0) {
                blk_max_val[b * (int)gridDim.x + (int)blockIdx.x] = bv;
                blk_max_idx[b * (int)gridDim.x + (int)blockIdx.x] = bi;
            }
        }
    }
}

__global__ void lm_head_stage2_argmax_kernel(
    const float* __restrict__ blk_max_val,
    const int32_t* __restrict__ blk_max_idx,
    const float* __restrict__ out_logits,
    int32_t* __restrict__ out_tokens,
    const SampleConfig* __restrict__ sample_cfg,
    const int32_t* __restrict__ d_pos,
    const int32_t* __restrict__ d_draft_tokens,
    const float* __restrict__ d_draft_probs,
    const float* __restrict__ d_draft_q_sel,
    const int32_t* __restrict__ d_draft_is_delta,
    const int32_t* __restrict__ inv_draft_vocab,
    int draft_vocab_size,
    int V,
    int B,
    int n_blocks) {
    pdl_wait_fenced(0u);
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    float bv = -1e30f;
    int32_t bi = 0;
    for (int i = tid; i < n_blocks; i += 256) {
        float v = ld_dep_f32(blk_max_val + b * n_blocks + i);
        int32_t idx = blk_max_idx[b * n_blocks + i];
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    __syncthreads();

    const bool do_sample = (sample_cfg != nullptr && sample_cfg->inv_temp > 0.0f && sample_cfg->top_k != 1 && out_logits != nullptr);
    if (!do_sample) {
        if (tid == 0) {
            bv = sv[0]; bi = si[0];
            for (int w = 1; w < 8; ++w) {
                if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
            }
            out_tokens[b] = bi;
        }
        return;
    }

    __shared__ float s_max_val;
    __shared__ int32_t s_max_idx;
    __shared__ unsigned long long hist_sum[256];
    __shared__ uint32_t hist_cnt[256];
    __shared__ float s_tau;
    __shared__ float s_inv_z;
    __shared__ int32_t s_accept;
    __shared__ int32_t s_draft_tok;
    __shared__ int32_t s_is_delta;

    if (tid == 0) {
        bv = sv[0]; bi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        s_max_val = bv;
        s_max_idx = bi;
    }
    __syncthreads();

    const float max_logit = s_max_val;
    const int32_t greedy_idx = s_max_idx;
    const float inv_temp = sample_cfg->inv_temp;
    const float top_p = sample_cfg->top_p;
    const int32_t top_k = sample_cfg->top_k;
    const uint64_t seed = sample_cfg->seed;
    const int32_t pos = d_pos ? (*d_pos + b) : b;
    const float* logits = out_logits + (size_t)b * V;

    constexpr float HIST_RANGE = 20.0f;
    constexpr float HIST_SCALE = 256.0f / HIST_RANGE; // 12.8 bins/nat

    if ((top_p > 0.0f && top_p < 0.9999f) || top_k > 0) {
        hist_sum[tid] = 0ULL;
        hist_cnt[tid] = 0u;
        __syncthreads();

        for (int row = tid; row < V; row += 256) {
            float d = (max_logit - ld_dep_f32(logits + row)) * inv_temp;
            if (d < HIST_RANGE) {
                int bin = (int)(d * HIST_SCALE);
                if (bin < 0) bin = 0;
                else if (bin > 255) bin = 255;
                unsigned long long fp = (unsigned long long)(__expf(-d) * 16777216.0f + 0.5f);
                atomicAdd(&hist_sum[bin], fp);
                atomicAdd(&hist_cnt[bin], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            double total_z = 0.0;
            for (int k = 0; k < 256; ++k) total_z += (double)hist_sum[k];
            double cum_z = 0.0;
            uint32_t cum_c = 0u;
            int cut_bin = 255;
            const double target_z = (top_p > 0.0f && top_p < 1.0f) ? ((double)top_p * total_z) : total_z;
            for (int k = 0; k < 256; ++k) {
                cum_z += (double)hist_sum[k];
                cum_c += hist_cnt[k];
                const bool hit_p = (top_p > 0.0f && top_p < 0.9999f && cum_z >= target_z);
                const bool hit_k = (top_k > 0 && cum_c >= (uint32_t)top_k);
                if (hit_p || hit_k) {
                    cut_bin = k;
                    break;
                }
            }
            s_tau = max_logit - ((float)(cut_bin + 1) / HIST_SCALE) / inv_temp;
        }
        __syncthreads();
    } else {
        if (tid == 0) s_tau = max_logit - HIST_RANGE / inv_temp;
        __syncthreads();
    }

    const float tau = s_tau;

    // Exact Leviathan/Chen stochastic rejection sampling for verified draft slots b in 0..B-2!
    if (d_draft_tokens != nullptr && b < B - 1) {
        float local_z = 0.0f;
        for (int row = tid; row < V; row += 256) {
            float l = ld_dep_f32(logits + row);
            if (l >= tau) local_z += __expf((l - max_logit) * inv_temp);
        }
        local_z = warp_sum(local_z);
        if ((tid & 31) == 0) sv[tid >> 5] = local_z;
        __syncthreads();
        if (tid == 0) {
            float tot_z = 0.0f;
            for (int w = 0; w < 8; ++w) tot_z += sv[w];
            float inv_z = (tot_z > 0.0f) ? (1.0f / tot_z) : 0.0f;
            s_inv_z = inv_z;
            int32_t x = d_draft_tokens[b];
            s_draft_tok = x;
            int32_t is_d = (d_draft_is_delta != nullptr && *d_draft_is_delta != 0) ? 1 : 0;
            s_is_delta = is_d;
            float lx = (x >= 0 && x < V) ? ld_dep_f32(logits + x) : -1e30f;
            float px = (lx >= tau) ? (__expf((lx - max_logit) * inv_temp) * inv_z) : 0.0f;
            float qx = is_d ? 1.0f : (d_draft_q_sel ? d_draft_q_sel[b] : 1.0f);
            float u_acc = hash_u01(seed ^ 0x517CC1B727220A95ULL, pos, 0x5A5A5A5A);
            int32_t acc = (px > 0.0f && (qx <= 0.0f || (u_acc * qx <= px))) ? 1 : 0;
            s_accept = acc;
            if (acc) out_tokens[b] = x;
        }
        __syncthreads();
        if (s_accept) return;

        // Rejected: exact residual resampling from norm(max(0, p(v) - q(v))) via Gumbel-Max!
        const float inv_z = s_inv_z;
        const int32_t x = s_draft_tok;
        const int32_t is_d = s_is_delta;
        float best_s = -1e30f;
        int32_t best_t = -1;
        for (int row = tid; row < V; row += 256) {
            float l = ld_dep_f32(logits + row);
            if (l >= tau) {
                float pv = __expf((l - max_logit) * inv_temp) * inv_z;
                float qv = 0.0f;
                if (is_d) {
                    qv = (row == x) ? 1.0f : 0.0f;
                } else if (d_draft_probs != nullptr) {
                    int32_t didx = inv_draft_vocab ? inv_draft_vocab[row] : row;
                    if (didx >= 0 && didx < draft_vocab_size) {
                        qv = d_draft_probs[(size_t)b * draft_vocab_size + didx];
                    }
                }
                float diff = pv - qv;
                if (diff > 0.0f) {
                    float sc = __logf(diff) + hash_gumbel(seed, pos, row);
                    if (sc > best_s) { best_s = sc; best_t = row; }
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float os = __shfl_xor_sync(0xffffffffu, best_s, off);
            int32_t ot = __shfl_xor_sync(0xffffffffu, best_t, off);
            if (os > best_s || (os == best_s && ot < best_t)) { best_s = os; best_t = ot; }
        }
        if ((tid & 31) == 0) { sv[tid >> 5] = best_s; si[tid >> 5] = best_t; }
        __syncthreads();
        if (tid == 0) {
            best_s = sv[0]; best_t = si[0];
            for (int w = 1; w < 8; ++w) {
                if (sv[w] > best_s || (sv[w] == best_s && si[w] < best_t)) { best_s = sv[w]; best_t = si[w]; }
            }
            // Never emit rejected draft token x on degeneracy fallback
            int32_t fallback = (greedy_idx != x) ? greedy_idx : ((x > 0) ? (x - 1) : 1);
            out_tokens[b] = (best_t >= 0) ? best_t : fallback;
        }
        return;
    }

    // Direct exact Gumbel-Max sampling from target p(v) (prefill, B=1, or speculative bonus token b=B-1)
    float best_s = -1e30f;
    int32_t best_t = greedy_idx;
    for (int row = tid; row < V; row += 256) {
        float l = ld_dep_f32(logits + row);
        if (l >= tau) {
            float sc = (l - max_logit) * inv_temp + hash_gumbel(seed, pos, row);
            if (sc > best_s) { best_s = sc; best_t = row; }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float os = __shfl_xor_sync(0xffffffffu, best_s, off);
        int32_t ot = __shfl_xor_sync(0xffffffffu, best_t, off);
        if (os > best_s || (os == best_s && ot < best_t)) { best_s = os; best_t = ot; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = best_s; si[tid >> 5] = best_t; }
    __syncthreads();
    if (tid == 0) {
        best_s = sv[0]; best_t = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > best_s || (sv[w] == best_s && si[w] < best_t)) { best_s = sv[w]; best_t = si[w]; }
        }
        out_tokens[b] = best_t;
    }
}

// ============================================================================
// ZERO-SMEM, ZERO-SPILL MoE Expert Kernels (B=1 AND B=2..8 Specialized Grouped!)
// ============================================================================

__global__ void build_expert_groups_kernel(
    const int32_t* __restrict__ route_ids,
    const float* __restrict__ route_w,
    ExpertGroups* __restrict__ g,
    int B) {
    if (threadIdx.x != 0) return;
    int n_uniq = 0;
    int exp_to_u[NUM_EXPERTS];
    for (int e = 0; e < NUM_EXPERTS; ++e) exp_to_u[e] = -1;
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < TOP_K; ++s) {
            int e = route_ids[b * TOP_K + s];
            float rw = route_w[b * TOP_K + s];
            int u = exp_to_u[e];
            if (u < 0) {
                u = n_uniq++;
                exp_to_u[e] = u;
                g->expert_id[u] = e;
                g->token_mask[u] = 0;
            }
            g->token_mask[u] |= uint8_t(1u << b);
            g->slot_idx[u][b] = uint8_t(s);
            g->route_w[u][b] = rw;
        }
    }
    g->num_unique = n_uniq;
}

// ============================================================================
// Blackwell Tensor Core (mma.m16n8k16) Decode MoE Kernels (M = 1..8)
// Register-permuted K loads + PDL overlap + 144-CTA fused multi-expert moe_down!
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0).
// ============================================================================

constexpr int MOE_TILE = 16;
constexpr int MOE_MAXM = 8;
constexpr int MOE_UP_WARPS = HIDDEN / 128;         // 18 warps (576 threads)
constexpr int MOE_DOWN_GROUPS = INTERMEDIATE / 128; // 9 column groups
constexpr int MOE_DOWN_PAR = 2;                     // 2 experts per round
constexpr int MOE_DOWN_WARPS = MOE_DOWN_GROUPS * MOE_DOWN_PAR; // 18 warps (576 threads)

template <int FMT> struct MoeFrag;
template <> struct MoeFrag<0> { Nvfp4Chunk32 r0, r1; uint32_t x[16]; };
template <> struct MoeFrag<1> { Mxfp8Chunk32 r0, r1; uint32_t x[16]; };

template <int FMT, int K>
__device__ __forceinline__ void moe_load_w(
    MoeFrag<FMT>& f, const uint8_t* codes, const uint8_t* scales, int c) {
    if constexpr (FMT == 0) {
        constexpr int CB = K >> 1, SB = K >> 4;
        f.r0 = load_nvfp4_chunk32(codes, scales, c);
        f.r1 = load_nvfp4_chunk32(codes + 8 * CB, scales + 8 * SB, c);
    } else {
        constexpr int CB = K, SB = K >> 5;
        f.r0 = load_mxfp8_chunk32(codes, scales, c);
        f.r1 = load_mxfp8_chunk32(codes + 8 * CB, scales + 8 * SB, c);
    }
}

template <int FMT>
__device__ __forceinline__ void moe_load_x(MoeFrag<FMT>& f, const __half* xrow, int c) {
    if (xrow) {
        const __half* xp = xrow + c * 32;
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            uint4 q = ld_dep_u4(xp + v * 8);
            if constexpr (FMT == 0) {
                f.x[4 * v + 0] = q.x; f.x[4 * v + 1] = q.y;
                f.x[4 * v + 2] = q.z; f.x[4 * v + 3] = q.w;
            } else {
                const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fv = __half22float2(*reinterpret_cast<const __half2*>(&ws[j]));
                    f.x[4 * v + j] = pack_bf16x2(fv.x, fv.y);
                }
            }
        }
    } else {
        #pragma unroll
        for (int v = 0; v < 16; ++v) f.x[v] = 0;
    }
}

template <int FMT>
__device__ __forceinline__ void moe_mma_frag(const MoeFrag<FMT>& f, float (&d)[4]) {
    uint32_t a[16], b[16];
    if constexpr (FMT == 0) {
        decode_nvfp4_chunk32_f16(f.r0, a);
        decode_nvfp4_chunk32_f16(f.r1, b);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            mma_m16n8k16_f32_f16(d[0], d[1], d[2], d[3],
                                 a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                 f.x[2 * i], f.x[2 * i + 1],
                                 d[0], d[1], d[2], d[3]);
        }
    } else {
        decode_mxfp8_chunk32_bf16(f.r0, a);
        decode_mxfp8_chunk32_bf16(f.r1, b);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            mma_m16n8k16_f32_bf16(d[0], d[1], d[2], d[3],
                                  a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1],
                                  f.x[2 * i], f.x[2 * i + 1],
                                  d[0], d[1], d[2], d[3]);
        }
    }
}

__device__ __forceinline__ uint64_t moe_slot_mask(int e, int lo, int hi) {
    uint32_t m0 = __ballot_sync(0xffffffffu, lo == e);
    uint32_t m1 = __ballot_sync(0xffffffffu, hi == e);
    return uint64_t(m0) | (uint64_t(m1) << 32);
}

template <int FMT>
__global__ void __launch_bounds__(MOE_UP_WARPS * 32)
moe_up_tc_kernel(
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ idx,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ ws,
    const float* __restrict__ wfactor,
    int M,
    __half* __restrict__ out_act_h) {
    constexpr int K = HIDDEN;
    constexpr int R = 2 * INTERMEDIATE;
    constexpr int CB = (FMT == 0) ? (K >> 1) : K;
    constexpr int SB = (FMT == 0) ? (K >> 4) : (K >> 5);
    __shared__ float red[MOE_UP_WARPS][32][4];
    __shared__ float tile_out[MOE_TILE][MOE_MAXM + 1];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int tile = blockIdx.x;
    const int slot = blockIdx.y;
    const int nslots = M * TOP_K;

    pdl_wait();
    pdl_trigger();

    const int lo = (lane < nslots) ? ld_dep_i32(idx + lane) : -1;
    const int hi = (lane + 32 < nslots) ? ld_dep_i32(idx + lane + 32) : -1;
    const int e = ld_dep_i32(idx + slot);
    const uint64_t mask = moe_slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;

    const size_t row = (size_t)e * R + tile * MOE_TILE + g;
    const bool on = (g < M) && (((mask >> (8 * g)) & 0xffu) != 0);
    MoeFrag<FMT> f;
    moe_load_w<FMT, K>(f, w + row * CB, ws + row * SB, 4 * warp + t);
    moe_load_x<FMT>(f, on ? (x_h + (size_t)g * K) : nullptr, 4 * warp + t);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    moe_mma_frag<FMT>(f, d);

    #pragma unroll
    for (int v = 0; v < 4; ++v) red[warp][lane][v] = d[v];
    __syncthreads();

    if (threadIdx.x < 32) {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int ww = 0; ww < MOE_UP_WARPS; ++ww) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) s[v] += red[ww][lane][v];
        }
        float es = (FMT == 0) ? wfactor[e] : 1.0f;
        tile_out[g][2 * t]         = s[0] * es;
        tile_out[g][2 * t + 1]     = s[1] * es;
        tile_out[g + 8][2 * t]     = s[2] * es;
        tile_out[g + 8][2 * t + 1] = s[3] * es;
    }
    __syncthreads();

    if (threadIdx.x < (MOE_TILE / 2) * MOE_MAXM) {
        const int p = threadIdx.x & 7;
        const int n = threadIdx.x >> 3;
        uint32_t kmask = (n < M) ? (uint32_t(mask >> (8 * n)) & 0xffu) : 0u;
        if (kmask) {
            const float gate = tile_out[2 * p][n];
            const float up   = tile_out[2 * p + 1][n];
            const float g_cap = fast_tanh(gate * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
            const float u_cap = fast_tanh(up * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
            const __half ah = __float2half((g_cap * fast_sigmoid(gate)) * u_cap);
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                out_act_h[((size_t)n * TOP_K + k) * INTERMEDIATE + tile * (MOE_TILE / 2) + p] = ah;
            }
        }
    }
}

template <int FMT>
__global__ void __launch_bounds__(MOE_DOWN_WARPS * 32)
moe_down_tc_kernel(
    const __half* __restrict__ act_h,
    const int32_t* __restrict__ idx,
    const float* __restrict__ wts,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ ws,
    const float* __restrict__ wfactor,
    int M,
    float* __restrict__ out_down) {
    constexpr int K = INTERMEDIATE;
    constexpr int R = HIDDEN;
    constexpr int CB = (FMT == 0) ? (K >> 1) : K;
    constexpr int SB = (FMT == 0) ? (K >> 4) : (K >> 5);
    __shared__ float part[MOE_DOWN_WARPS][32][4];
    __shared__ float ybuf[TOP_K * MOE_MAXM][MOE_TILE][MOE_MAXM];
    __shared__ int uexp[TOP_K * MOE_MAXM];
    __shared__ unsigned long long umask[TOP_K * MOE_MAXM];
    __shared__ uint8_t slot_u[TOP_K * MOE_MAXM];
    __shared__ int nuniq;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int tile = blockIdx.x;
    const int nslots = M * TOP_K;

    // idx was completed by router before moe_up passed its wait and triggered us!
    if (warp == 0) {
        const int lo = (lane < nslots) ? ld_dep_i32(idx + lane) : -1;
        const int hi = (lane + 32 < nslots) ? ld_dep_i32(idx + lane + 32) : -1;
        uint64_t remaining = (nslots == 64) ? ~0ull : ((1ull << nslots) - 1);
        int u = 0;
        for (; remaining; ++u) {
            const int s = __ffsll(static_cast<long long>(remaining)) - 1;
            const int e = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
            const uint64_t mask = moe_slot_mask(e, lo, hi);
            remaining &= ~mask;
            if (lane == 0) { uexp[u] = e; umask[u] = mask; }
            if ((mask >> lane) & 1) slot_u[lane] = uint8_t(u);
            if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(u);
        }
        if (lane == 0) nuniq = u;
    }
    __syncthreads();

    const int U = nuniq;
    const int p = warp / MOE_DOWN_GROUPS;
    const int c = 4 * (warp % MOE_DOWN_GROUPS) + t;

    auto xrow = [&](int u) -> const __half* {
        const uint32_t kmask = (g < M) ? (uint32_t(umask[u] >> (8 * g)) & 0xffu) : 0u;
        return kmask ? (act_h + ((size_t)g * TOP_K + (__ffs(kmask) - 1)) * K) : nullptr;
    };
    auto load_w_u = [&](int u, MoeFrag<FMT>& f) {
        const int e = uexp[u];
        const size_t row = (size_t)e * R + tile * MOE_TILE + g;
        moe_load_w<FMT, K>(f, w + row * CB, ws + row * SB, c);
    };

    // Prefetch round 0 W2 weights BEFORE pdl_wait() -> hides moe_up tail completely!
    MoeFrag<FMT> nxt;
    if (p < U) load_w_u(p, nxt);
    pdl_wait();
    if (p < U) moe_load_x<FMT>(nxt, xrow(p), c);

    #pragma unroll 1
    for (int base = 0; base < U; base += MOE_DOWN_PAR) {
        const MoeFrag<FMT> cur = nxt;
        const int u = base + p;
        if (u + MOE_DOWN_PAR < U) {
            load_w_u(u + MOE_DOWN_PAR, nxt);
            moe_load_x<FMT>(nxt, xrow(u + MOE_DOWN_PAR), c);
        }
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        if (u < U) moe_mma_frag<FMT>(cur, d);
        #pragma unroll
        for (int v = 0; v < 4; ++v) part[warp][lane][v] = d[v];
        __syncthreads();

        if (threadIdx.x < MOE_DOWN_PAR * 32 && base + (threadIdx.x >> 5) < U) {
            const int pp = threadIdx.x >> 5;
            float s[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int j = 0; j < MOE_DOWN_GROUPS; ++j) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) s[v] += part[pp * MOE_DOWN_GROUPS + j][lane][v];
            }
            const float es = (FMT == 0) ? wfactor[uexp[base + pp]] : 1.0f;
            float (&y)[MOE_TILE][MOE_MAXM] = ybuf[base + pp];
            y[g][2 * t]         = s[0] * es;
            y[g][2 * t + 1]     = s[1] * es;
            y[g + 8][2 * t]     = s[2] * es;
            y[g + 8][2 * t + 1] = s[3] * es;
        }
        __syncthreads();
    }
    pdl_trigger();

    if (threadIdx.x < MOE_TILE * M) {
        const int r = threadIdx.x & (MOE_TILE - 1);
        const int n = threadIdx.x / MOE_TILE;
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            const int sl = n * TOP_K + k;
            s = fmaf(ld_dep_f32(wts + sl), ybuf[slot_u[sl]][r][n], s);
        }
        out_down[(size_t)n * HIDDEN + tile * MOE_TILE + r] = s;
    }
}

} // namespace

void launch_f32_to_f16(const float* src, __half* dst, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((n + 3) / 4 + threads - 1) / threads;
    f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void launch_linear(
    const LinearWeight& w, const float* x, float* y, int B, bool add_residual, cudaStream_t stream) {
    while (B > 0) {
        int chunk = (B >= 8) ? 8 : B;
        switch (chunk) {
            case 1: dispatch_linear_b<1>(w, x, y, add_residual, stream); break;
            case 2: dispatch_linear_b<2>(w, x, y, add_residual, stream); break;
            case 3: dispatch_linear_b<3>(w, x, y, add_residual, stream); break;
            case 4: dispatch_linear_b<4>(w, x, y, add_residual, stream); break;
            case 5: dispatch_linear_b<5>(w, x, y, add_residual, stream); break;
            case 6: dispatch_linear_b<6>(w, x, y, add_residual, stream); break;
            case 7: dispatch_linear_b<7>(w, x, y, add_residual, stream); break;
            case 8: dispatch_linear_b<8>(w, x, y, add_residual, stream); break;
        }
        x += (size_t)chunk * w.cols;
        y += (size_t)chunk * w.rows;
        B -= chunk;
    }
}

void launch_linear_h(
    const LinearWeight& w, const __half* x_h, float* y, int B, bool add_residual, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_linear_h_b<1>(w, x_h, y, add_residual, stream); break;
        case 2: dispatch_linear_h_b<2>(w, x_h, y, add_residual, stream); break;
        case 3: dispatch_linear_h_b<3>(w, x_h, y, add_residual, stream); break;
        case 4: dispatch_linear_h_b<4>(w, x_h, y, add_residual, stream); break;
        case 5: dispatch_linear_h_b<5>(w, x_h, y, add_residual, stream); break;
        case 6: dispatch_linear_h_b<6>(w, x_h, y, add_residual, stream); break;
        case 7: dispatch_linear_h_b<7>(w, x_h, y, add_residual, stream); break;
        case 8: dispatch_linear_h_b<8>(w, x_h, y, add_residual, stream); break;
    }
}

template <int B_CONST>
static void dispatch_qkvrg_b(
    const LayerWeights& lw, const __half* xq, const __half* xk, const __half* xv,
    float* y_qrgs, float* y_k, float* y_v, cudaStream_t stream) {
    if (lw.w_qrgs.kind == WeightKind::FP8) {
        if constexpr (B_CONST == 1) {
            constexpr int N_TOT = (Q_DIM + R_DIM + Q_DIM + NUM_HEADS) + 2 * KV_DIM; // 8600
            int grid = (N_TOT + 7) / 8;
            launch_pdl(qkvrg_fused_fp8_b1_pdl_kernel, dim3(grid), 128, stream,
                static_cast<const uint8_t*>(lw.w_qrgs.data), lw.w_qrgs.row_scales,
                static_cast<const uint8_t*>(lw.w_k.data), lw.w_k.row_scales,
                static_cast<const uint8_t*>(lw.w_v.data), lw.w_v.row_scales,
                xq, xk, xv, y_qrgs, y_k, y_v, 0u);
        } else {
            constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552 -> 410 tiles
            constexpr int GRID = ((N_Q + 15) / 16) + (KV_DIM / 16) + (KV_DIM / 16); // 538 CTAs
            launch_pdl(qkvrg_fused_fp8_mma_kernel<B_CONST>, dim3(GRID), 192, stream,
                static_cast<const uint8_t*>(lw.w_qrgs.data), lw.w_qrgs.row_scales,
                static_cast<const uint8_t*>(lw.w_k.data), lw.w_k.row_scales,
                static_cast<const uint8_t*>(lw.w_v.data), lw.w_v.row_scales,
                xq, xk, xv, y_qrgs, y_k, y_v, 0u);
        }
        return;
    }
    if constexpr (B_CONST == 1) {
        constexpr int N_TOT = (Q_DIM + R_DIM + Q_DIM + NUM_HEADS) + 2 * KV_DIM; // 8600
        int grid = (N_TOT + 7) / 8; // 1075 half-warp blocks!
        launch_pdl(qkvrg_fused_nvfp4_b1_pdl_kernel, dim3(grid), 128, stream,
            static_cast<const uint8_t*>(lw.w_qrgs.data), lw.w_qrgs.scales, lw.w_qrgs.global_scale,
            static_cast<const uint8_t*>(lw.w_k.data), lw.w_k.scales, lw.w_k.global_scale,
            static_cast<const uint8_t*>(lw.w_v.data), lw.w_v.scales, lw.w_v.global_scale,
            xq, xk, xv, y_qrgs, y_k, y_v, 0u);
    } else {
        constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552 -> 410 tiles
        constexpr int GRID = ((N_Q + 15) / 16) + (KV_DIM / 16) + (KV_DIM / 16); // 538 CTAs
        launch_pdl(qkvrg_fused_nvfp4_mma_kernel<B_CONST>, dim3(GRID), 192, stream,
            static_cast<const uint8_t*>(lw.w_qrgs.data), lw.w_qrgs.scales, lw.w_qrgs.global_scale,
            static_cast<const uint8_t*>(lw.w_k.data), lw.w_k.scales, lw.w_k.global_scale,
            static_cast<const uint8_t*>(lw.w_v.data), lw.w_v.scales, lw.w_v.global_scale,
            xq, xk, xv, y_qrgs, y_k, y_v, 0u);
    }
}

void launch_qkvrg_fused(
    const LayerWeights& lw, const __half* in_q_h, const __half* in_k_h, const __half* in_v_h,
    float* qrgs_out, float* k_out, float* v_out, int B, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_qkvrg_b<1>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 2: dispatch_qkvrg_b<2>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 3: dispatch_qkvrg_b<3>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 4: dispatch_qkvrg_b<4>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 5: dispatch_qkvrg_b<5>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 6: dispatch_qkvrg_b<6>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 7: dispatch_qkvrg_b<7>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 8: dispatch_qkvrg_b<8>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
    }
}

void launch_bf16_linear(
    const __nv_bfloat16* w, const float* x, float* y, int B, int N, int K, cudaStream_t stream) {
    LinearWeight lw;
    lw.kind = WeightKind::BF16;
    lw.rows = N;
    lw.cols = K;
    lw.data = const_cast<__nv_bfloat16*>(w);
    launch_linear(lw, x, y, B, false, stream);
}

template <int B_CONST>
static void dispatch_lm_head(
    const LinearWeight& u, const float* x, int32_t* out_toks, float* out_logits,
    float* s_val, int32_t* s_idx, const SampleConfig* sample_cfg, const int32_t* d_pos,
    const int32_t* d_draft_tokens, const float* d_draft_probs, const float* d_draft_q_sel,
    const int32_t* d_draft_is_delta, const int32_t* inv_draft_vocab, int draft_vocab_size,
    cudaStream_t stream) {
    const int lm_blocks = get_lm_head_blocks();
    if (u.kind == WeightKind::NVFP4) {
        if constexpr (B_CONST >= 2) {
            launch_pdl(lm_head_nvfp4_mma_stage1_kernel<B_CONST>, dim3(lm_blocks), 192, stream,
                static_cast<const uint8_t*>(u.data), u.scales, u.global_scale,
                x, out_logits, s_val, s_idx, sample_cfg, u.rows, 0u);
        } else {
            launch_pdl(lm_head_stage1_kernel<B_CONST, 2>, dim3(lm_blocks), 256, stream,
                u.data, u.scales, static_cast<const float*>(nullptr), u.global_scale,
                x, out_logits, s_val, s_idx, sample_cfg, u.rows);
        }
    } else if (u.kind == WeightKind::FP8) {
        if constexpr (B_CONST >= 2) {
            launch_pdl(lm_head_fp8_mma_stage1_kernel<B_CONST>, dim3(lm_blocks), 192, stream,
                static_cast<const uint8_t*>(u.data), u.row_scales,
                x, out_logits, s_val, s_idx, sample_cfg, u.rows, 0u);
        } else {
            launch_pdl(lm_head_stage1_kernel<B_CONST, 1>, dim3(lm_blocks), 256, stream,
                u.data, static_cast<const uint8_t*>(nullptr), u.row_scales, 1.0f,
                x, out_logits, s_val, s_idx, sample_cfg, u.rows);
        }
    } else {
        lm_head_stage1_kernel<B_CONST, 0><<<lm_blocks, 256, 0, stream>>>(
            u.data, nullptr, nullptr, 1.0f, x, out_logits, s_val, s_idx, sample_cfg, u.rows);
    }
    launch_pdl(lm_head_stage2_argmax_kernel, dim3(B_CONST), 256, stream,
        s_val, s_idx, out_logits, out_toks, sample_cfg, d_pos,
        d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta,
        inv_draft_vocab, draft_vocab_size, u.rows, B_CONST, lm_blocks);
}

void launch_lm_head_argmax(
    const LinearWeight& unembed, const float* x, int32_t* out_tokens, float* out_logits,
    float* scratch_vals, int32_t* scratch_idxs, const SampleConfig* sample_cfg,
    const int32_t* d_pos, const int32_t* d_draft_tokens, const float* d_draft_probs,
    const float* d_draft_q_sel, const int32_t* d_draft_is_delta, const int32_t* inv_draft_vocab,
    int draft_vocab_size, int B, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_lm_head<1>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 2: dispatch_lm_head<2>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 3: dispatch_lm_head<3>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 4: dispatch_lm_head<4>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 5: dispatch_lm_head<5>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 6: dispatch_lm_head<6>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 7: dispatch_lm_head<7>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        case 8: dispatch_lm_head<8>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, sample_cfg, d_pos, d_draft_tokens, d_draft_probs, d_draft_q_sel, d_draft_is_delta, inv_draft_vocab, draft_vocab_size, stream); break;
        default:
            for (int i = 0; i < B; ++i) {
                dispatch_lm_head<1>(unembed, x + (size_t)i * HIDDEN, out_tokens + i,
                    out_logits ? out_logits + (size_t)i * unembed.rows : nullptr,
                    scratch_vals, scratch_idxs, sample_cfg, d_pos,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, stream);
            }
            break;
    }
}

void launch_build_expert_groups(
    const int32_t* route_ids, const float* route_w, ExpertGroups* groups, int B, cudaStream_t stream) {
    build_expert_groups_kernel<<<1, 1, 0, stream>>>(route_ids, route_w, groups, B);
}

void launch_moe_gate_up_h(
    const ExpertBank& w13, const __half* x_h, const int32_t* route_ids,
    const ExpertGroups* /*groups*/, __half* out_act_h, int B, cudaStream_t stream) {
    dim3 grid((2 * INTERMEDIATE) / MOE_TILE, B * TOP_K);
    if (w13.is_nvfp4) {
        launch_pdl(moe_up_tc_kernel<0>, grid, MOE_UP_WARPS * 32, stream,
            x_h, route_ids, w13.data, w13.scales, w13.expert_scale, B, out_act_h);
    } else {
        launch_pdl(moe_up_tc_kernel<1>, grid, MOE_UP_WARPS * 32, stream,
            x_h, route_ids, w13.data, w13.scales, static_cast<const float*>(nullptr), B, out_act_h);
    }
}

void launch_moe_down_h(
    const ExpertBank& w2, const __half* act_h, const int32_t* route_ids, const float* route_w,
    const ExpertGroups* /*groups*/, float* out_down, int B, cudaStream_t stream, bool /*already_zeroed*/) {
    dim3 grid(HIDDEN / MOE_TILE);
    if (w2.is_nvfp4) {
        launch_pdl(moe_down_tc_kernel<0>, grid, MOE_DOWN_WARPS * 32, stream,
            act_h, route_ids, route_w, w2.data, w2.scales, w2.expert_scale, B, out_down);
    } else {
        launch_pdl(moe_down_tc_kernel<1>, grid, MOE_DOWN_WARPS * 32, stream,
            act_h, route_ids, route_w, w2.data, w2.scales, static_cast<const float*>(nullptr), B, out_down);
    }
}

} // namespace squid

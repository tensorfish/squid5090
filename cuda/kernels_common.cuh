#pragma once

#include "config.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <utility>

namespace squid {

// ============================================================================
// Blackwell Programmatic Dependent Launch (PDL) & Streaming Weight Load Primitives
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0).
// Note: L2::cache_hint is intentionally omitted on CUDA 13.1 due to ptxas 13.1 bug.
// ============================================================================

__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_wait_fenced(uint32_t zero) {
#if __CUDA_ARCH__ >= 900
    #pragma unroll 1
    for (uint32_t i = 0; i <= zero; ++i) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#else
    (void)zero;
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <class... KArgs, class... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, int block, cudaStream_t stream, Args&&... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

__device__ __forceinline__ uint4 ld_stream_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_stream_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint16_t ld_scale_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint8_t ld_scale_u8(const void* p) {
    uint16_t v;
    asm volatile("ld.global.nc.u8 %0, [%1];" : "=h"(v) : "l"(p));
    return static_cast<uint8_t>(v);
}

// Coherent, non-hoistable loads for activations/indices produced by a PDL predecessor.
__device__ __forceinline__ uint4 ld_dep_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ int32_t ld_dep_i32(const int32_t* p) {
    int32_t v;
    asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ float ld_dep_f32(const float* p) {
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ float4 ld_dep_f4(const float* p) {
    float4 v;
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void mma_m16n8k16_f32_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// Hardware Blackwell sm_120 PTX FP4 (e2m1x2) -> half2 conversion!
// Single-instruction hardware decode on Blackwell (zero LUT / bitshift overhead).
__device__ __forceinline__ __half2 cvt_e2m1x2_to_half2(uint32_t byte_val) {
    uint32_t out;
#if __CUDA_ARCH__ >= 1000
    uint16_t tmp = static_cast<uint16_t>(byte_val);
    asm("{ .reg .b8 b0, b1;\n"
        "  mov.b16 {b0, b1}, %1;\n"
        "  cvt.rn.f16x2.e2m1x2 %0, b0; }\n"
        : "=r"(out) : "h"(tmp));
#else
    static constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    __half2 h = __floats2half2_rn(lut[byte_val & 0xFu], lut[(byte_val >> 4) & 0xFu]);
    out = *reinterpret_cast<uint32_t*>(&h);
#endif
    return *reinterpret_cast<__half2*>(&out);
}

// Unpack 8 FP4 weights (1 uint32) into 4x half2 with ZERO shifts/masks on Blackwell sm_120!
__device__ __forceinline__ void cvt_e2m1x8_to_half2x4(
    uint32_t w, __half2& h0, __half2& h1, __half2& h2, __half2& h3) {
#if __CUDA_ARCH__ >= 1000
    uint32_t r0, r1, r2, r3;
    asm("{ .reg .b8 b0, b1, b2, b3;\n"
        "  mov.b32 {b0, b1, b2, b3}, %4;\n"
        "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %3, b3; }\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(w));
    h0 = *reinterpret_cast<__half2*>(&r0);
    h1 = *reinterpret_cast<__half2*>(&r1);
    h2 = *reinterpret_cast<__half2*>(&r2);
    h3 = *reinterpret_cast<__half2*>(&r3);
#else
    h0 = cvt_e2m1x2_to_half2(w & 0xFFu);
    h1 = cvt_e2m1x2_to_half2((w >> 8) & 0xFFu);
    h2 = cvt_e2m1x2_to_half2((w >> 16) & 0xFFu);
    h3 = cvt_e2m1x2_to_half2((w >> 24) & 0xFFu);
#endif
}

// Hardware Blackwell/Hopper/Ada FP8 E4M3x2 -> half2 conversion
__device__ __forceinline__ __half2 cvt_e4m3x2_to_half2(uint16_t packed) {
    uint32_t out;
#if __CUDA_ARCH__ >= 890
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(packed));
#else
    __nv_fp8x2_e4m3 v;
    v.__x = packed;
    __half2 h = static_cast<__half2>(v);
    out = *reinterpret_cast<uint32_t*>(&h);
#endif
    return *reinterpret_cast<__half2*>(&out);
}

__device__ __forceinline__ float cvt_e4m3_to_float(uint8_t code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<float>(v);
}

__device__ __forceinline__ uint8_t float_to_e4m3(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float cvt_ue8m0_to_float(uint8_t e) {
    uint32_t bits = (e == 0) ? 0x00400000u : (e == 255 ? 0x7fc00000u : (uint32_t(e) << 23));
    return __uint_as_float(bits);
}

__device__ __forceinline__ float bf16_to_float(uint16_t b) {
    return __uint_as_float(uint32_t(b) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&b);
}

__device__ __forceinline__ float round_bf16(float f) {
    return __bfloat162float(__float2bfloat16(f));
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float half_warp_sum(float v) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float quarter_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float half2_dot_f32(__half2 a, __half2 b) {
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    return fmaf(fa.x, fb.x, fa.y * fb.y);
}

// Compute dot product of 16 elements: 8 bytes (uint2) of packed e2m1 weights x 16 halfs (4x half2 pairs),
// accumulated in half2 and returned as float (to be multiplied by E4M3 block scale).
__device__ __forceinline__ float nvfp4_block16_dot(uint2 w8, const __half2* __restrict__ xh) {
    __half2 h0, h1, h2, h3, h4, h5, h6, h7;
    cvt_e2m1x8_to_half2x4(w8.x, h0, h1, h2, h3);
    cvt_e2m1x8_to_half2x4(w8.y, h4, h5, h6, h7);
    __half2 p0 = __hmul2(h0, xh[0]);
    __half2 p1 = __hmul2(h1, xh[1]);
    p0 = __hfma2(h2, xh[2], p0);
    p1 = __hfma2(h3, xh[3], p1);
    p0 = __hfma2(h4, xh[4], p0);
    p1 = __hfma2(h5, xh[5], p1);
    p0 = __hfma2(h6, xh[6], p0);
    p1 = __hfma2(h7, xh[7], p1);
    float2 f = __half22float2(__hadd2(p0, p1));
    return f.x + f.y;
}

// Pack two float values into half2
__device__ __forceinline__ __half2 f32_pair_to_h2(float a, float b) {
    return __floats2half2_rn(a, b);
}

// Hardware FP16 Tensor Core MMA: D[16,8] = A[16,16] * B[16,8] + C[16,8] (FP32 accumulate)
__device__ __forceinline__ void mma_m16n8k16_f32_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

// Split-K attention partial buffer offsets (NUM_HEADS=24, MAX_VERIFY_B=8, KV_TOTAL_SPLITS=65)
constexpr size_t SPLIT_MD_ELEMS = (size_t)NUM_HEADS * 8 * KV_TOTAL_SPLITS; // 12480
constexpr size_t SPLIT_M_OFF = 0;
constexpr size_t SPLIT_D_OFF = SPLIT_MD_ELEMS;
constexpr size_t SPLIT_O_OFF = 2 * SPLIT_MD_ELEMS; // 24960
constexpr size_t SPLIT_BUF_FLOATS = SPLIT_O_OFF + SPLIT_MD_ELEMS * HEAD_DIM; // 1622400

// Store one 128-dim KV head using 128 threads (tid = 0..127, 4 warps, caller provides sm[4]).
// KV_FMT: 0 = BF16, 1 = FP8 (E4M3 + per-head half scale), 2 = Q4 ([-7..7]+7 + group-64 half scale).
template <int KV_FMT>
__device__ __forceinline__ void store_kv_head_128(
    float val, int slot, int h, int tid,
    void* __restrict__ cache, __half* __restrict__ scales, float* sm) {
    if constexpr (KV_FMT == 0) {
        static_cast<__nv_bfloat16*>(cache)[(size_t)slot * KV_DIM + h * HEAD_DIM + tid] = __float2bfloat16(val);
    } else if constexpr (KV_FMT == 1) {
        int warp = tid >> 5;
        int lane = tid & 31;
        float wm = warp_max(fabsf(val));
        if (lane == 0) sm[warp] = wm;
        __syncthreads();
        float amax = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
        float sc = fmaxf(amax / 448.0f, 1e-7f);
        if (tid == 0) scales[(size_t)slot * NUM_KV_HEADS + h] = __float2half(sc);
        static_cast<uint8_t*>(cache)[(size_t)slot * KV_DIM + h * HEAD_DIM + tid] = float_to_e4m3(val / sc);
    } else {
        int warp = tid >> 5;
        int lane = tid & 31;
        float wm = warp_max(fabsf(val));
        if (lane == 0) sm[warp] = wm;
        __syncthreads();
        int g = tid >> 6; // 0 (dims 0..63) or 1 (dims 64..127)
        float gmax = fmaxf(sm[g * 2], sm[g * 2 + 1]);
        float sc = fmaxf(gmax / 7.0f, 1e-7f);
        if ((tid & 63) == 0) scales[(size_t)slot * KV_Q4_SCALES_PER_TOK + h * 2 + g] = __float2half(sc);
        int q = max(-7, min(7, __float2int_rn(val / sc))) + 7; // 0..14
        int q_odd = __shfl_xor_sync(0xffffffffu, q, 1);
        if ((tid & 1) == 0) {
            static_cast<uint8_t*>(cache)[(size_t)slot * (KV_DIM >> 1) + h * 64 + (tid >> 1)] =
                static_cast<uint8_t>(q | (q_odd << 4));
        }
    }
}

// Load 4 contiguous KV elements at (slot, kvh, d_off = lane*4) into two float2s.
template <int KV_FMT>
__device__ __forceinline__ void load_kv_4elems(
    const void* __restrict__ cache, const __half* __restrict__ scales,
    int slot, int kvh, int d_off, float2& out0, float2& out1) {
    if constexpr (KV_FMT == 0) {
        uint2 raw = __ldg(reinterpret_cast<const uint2*>(
            static_cast<const __nv_bfloat16*>(cache) + (size_t)slot * KV_DIM + kvh * HEAD_DIM + d_off));
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
        out0 = __bfloat1622float2(b[0]);
        out1 = __bfloat1622float2(b[1]);
    } else if constexpr (KV_FMT == 1) {
        uint32_t raw = __ldg(reinterpret_cast<const uint32_t*>(
            static_cast<const uint8_t*>(cache) + (size_t)slot * KV_DIM + kvh * HEAD_DIM + d_off));
        float sc = __half2float(__ldg(scales + (size_t)slot * NUM_KV_HEADS + kvh));
        float2 f0 = __half22float2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(raw & 0xFFFFu)));
        float2 f1 = __half22float2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(raw >> 16)));
        out0 = make_float2(f0.x * sc, f0.y * sc);
        out1 = make_float2(f1.x * sc, f1.y * sc);
    } else {
        uint16_t raw = __ldg(reinterpret_cast<const uint16_t*>(
            static_cast<const uint8_t*>(cache) + (size_t)slot * (KV_DIM >> 1) + kvh * 64 + (d_off >> 1)));
        float sc = __half2float(__ldg(scales + (size_t)slot * KV_Q4_SCALES_PER_TOK + kvh * 2 + (d_off >> 6)));
        out0 = make_float2((float(int(raw & 0xFu) - 7)) * sc, (float(int((raw >> 4) & 0xFu) - 7)) * sc);
        out1 = make_float2((float(int((raw >> 8) & 0xFu) - 7)) * sc, (float(int((raw >> 12) & 0xFu) - 7)) * sc);
    }
}

// Cooperative loader: load 8 contiguous KV elements at (slot, kvh, sub*8) into 4x half2 (128-bit uint4)!
template <int KV_FMT>
__device__ __forceinline__ uint4 load_kv_8elems_h2x4(
    const void* __restrict__ cache, const __half* __restrict__ scales,
    int slot, int kvh, int sub) {
    __half2 h[4];
    if constexpr (KV_FMT == 0) {
        uint4 raw = __ldg(reinterpret_cast<const uint4*>(
            static_cast<const __nv_bfloat16*>(cache) + (size_t)slot * KV_DIM + kvh * HEAD_DIM) + sub);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __bfloat1622float2(b[i]);
            h[i] = __floats2half2_rn(f.x, f.y);
        }
    } else if constexpr (KV_FMT == 1) {
        uint2 raw = __ldg(reinterpret_cast<const uint2*>(
            static_cast<const uint8_t*>(cache) + (size_t)slot * KV_DIM + kvh * HEAD_DIM) + sub);
        __half sc = __ldg(scales + (size_t)slot * NUM_KV_HEADS + kvh);
        __half2 hsc = __half2half2(sc);
        h[0] = __hmul2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(raw.x & 0xFFFFu)), hsc);
        h[1] = __hmul2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(raw.x >> 16)), hsc);
        h[2] = __hmul2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(raw.y & 0xFFFFu)), hsc);
        h[3] = __hmul2(cvt_e4m3x2_to_half2(static_cast<uint16_t>(raw.y >> 16)), hsc);
    } else {
        uint32_t raw = __ldg(reinterpret_cast<const uint32_t*>(
            static_cast<const uint8_t*>(cache) + (size_t)slot * (KV_DIM >> 1) + kvh * 64) + sub);
        __half sc = __ldg(scales + (size_t)slot * KV_Q4_SCALES_PER_TOK + kvh * 2 + (sub >> 3));
        __half2 hsc = __half2half2(sc);
        __half2 h1031 = __float2half2_rn(1031.0f);
        // Fast exact FP16 magic-bias nibble unpack: (0x6400 | n) = 1024+n -> sub 1031 = n-7 -> mul sc!
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t pair = (raw >> (i * 8)) & 0xFFu;
            uint32_t u32 = (0x6400u | (pair & 0xFu)) | ((0x6400u | (pair >> 4)) << 16);
            h[i] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&u32), h1031), hsc);
        }
    }
    return *reinterpret_cast<uint4*>(h);
}

// Shared Blackwell FP16 Tensor Core (m16n8k16 + ldmatrix.x4 / ldmatrix.x2.trans) Split-K Bulk Engine!
// TILE_T = 32 (2 slices x 16 tokens): 17.4 KB SMEM (s_k + s_v together -> ONLY 2 syncs/tile, 4-5 blocks/SM!).
// NM = 1 (THREADS=64) for Q_CNT <= 16 (B=1..5); NM = 2 (THREADS=128) for Q_CNT <= 32 (B=6..8, DSpark).
template <int KV_FMT, int Q_CNT, int Q_STRIDE, int THREADS>
__device__ __forceinline__ void run_tc_bulk_splitk(
    const float* __restrict__ q_base_ptr,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    int kv_cap,
    int kvh, int split_slot, int t_start, int t_end,
    float* __restrict__ split_buf) {
    constexpr int NM = (Q_CNT > 16) ? 2 : 1;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;   // 0..7
    int tid4 = lane & 3;   // 0..3
    int slice = warp & 1;  // 0..1 (2 warps split TILE_T=32 into 16 tokens each!)
    int m_tile = warp >> 1;// 0 (or 1 when NM==2)

    float* s_m_out = split_buf + SPLIT_M_OFF;
    float* s_d_out = split_buf + SPLIT_D_OFF;
    float* s_o_out = split_buf + SPLIT_O_OFF;

    if (t_start >= t_end) {
        if (tid < Q_CNT) {
            int gqa = tid % GQA_RATIO;
            int b = tid / GQA_RATIO;
            int h = kvh * GQA_RATIO + gqa;
            int idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
            s_m_out[idx] = -1e30f;
            s_d_out[idx] = 0.0f;
        }
        return;
    }

    int q0 = m_tile * 16 + grp;
    int q1 = m_tile * 16 + grp + 8;
    const float* qp0 = nullptr;
    const float* qp1 = nullptr;
    if (q0 < Q_CNT) {
        int h0 = kvh * GQA_RATIO + (q0 % GQA_RATIO);
        int b0 = q0 / GQA_RATIO;
        qp0 = q_base_ptr + (size_t)b0 * Q_STRIDE + h0 * HEAD_DIM;
    }
    if (q1 < Q_CNT) {
        int h1 = kvh * GQA_RATIO + (q1 % GQA_RATIO);
        int b1 = q1 / GQA_RATIO;
        qp1 = q_base_ptr + (size_t)b1 * Q_STRIDE + h1 * HEAD_DIM;
    }

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        int k_off = kt * 16;
        if (qp0) {
            __half2 h0 = __floats2half2_rn(qp0[k_off + tid4 * 2] * INV_SQRT_HEAD_DIM,
                                           qp0[k_off + tid4 * 2 + 1] * INV_SQRT_HEAD_DIM);
            __half2 h2 = __floats2half2_rn(qp0[k_off + tid4 * 2 + 8] * INV_SQRT_HEAD_DIM,
                                           qp0[k_off + tid4 * 2 + 9] * INV_SQRT_HEAD_DIM);
            q_reg[kt][0] = *reinterpret_cast<uint32_t*>(&h0);
            q_reg[kt][2] = *reinterpret_cast<uint32_t*>(&h2);
        } else {
            q_reg[kt][0] = 0u;
            q_reg[kt][2] = 0u;
        }
        if (qp1) {
            __half2 h1 = __floats2half2_rn(qp1[k_off + tid4 * 2] * INV_SQRT_HEAD_DIM,
                                           qp1[k_off + tid4 * 2 + 1] * INV_SQRT_HEAD_DIM);
            __half2 h3 = __floats2half2_rn(qp1[k_off + tid4 * 2 + 8] * INV_SQRT_HEAD_DIM,
                                           qp1[k_off + tid4 * 2 + 9] * INV_SQRT_HEAD_DIM);
            q_reg[kt][1] = *reinterpret_cast<uint32_t*>(&h1);
            q_reg[kt][3] = *reinterpret_cast<uint32_t*>(&h3);
        } else {
            q_reg[kt][1] = 0u;
            q_reg[kt][3] = 0u;
        }
    }

    float m0 = -1e30f, m1 = -1e30f;
    float d0 = 0.0f, d1 = 0.0f;
    float o[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    }

    // Only 17.4 KB SMEM with BOTH s_k AND s_v resident together -> 2 syncs/tile + 4-5 blocks/SM!
    __shared__ union {
        struct {
            __half s_k[32][HEAD_DIM + 8]; // 8,704 B
            __half s_v[32][HEAD_DIM + 8]; // 8,704 B
        } kv;
        struct {
            float m[16][2];
            float d[16][2];
            float o[2][16][HEAD_DIM];     // 16,384 B
        } red;
    } smem;

    int c0 = tid4 * 2;
    int c1 = tid4 * 2 + 1;
    int c4 = tid4 * 2 + 8;
    int c5 = tid4 * 2 + 9;
    int t_base = slice * 16;
    int k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
    int k_ld_col = (lane & 8);
    int v_ld_row = t_base + (lane & 15);

    for (int t0 = t_start; t0 < t_end; t0 += 32) {
        int tile_len = min(32, t_end - t0);
        // 1. Load K AND V together in one coalesced HBM burst (512 tasks: 32 tok x 16 sub)
        #pragma unroll
        for (int idx = tid; idx < 512; idx += THREADS) {
            int t_loc = idx >> 4;
            int sub = idx & 15;
            if (t_loc < tile_len) {
                int slot = (t0 + t_loc) % kv_cap;
                *reinterpret_cast<uint4*>(&smem.kv.s_k[t_loc][sub * 8]) =
                    load_kv_8elems_h2x4<KV_FMT>(k_cache, k_scale, slot, kvh, sub);
                *reinterpret_cast<uint4*>(&smem.kv.s_v[t_loc][sub * 8]) =
                    load_kv_8elems_h2x4<KV_FMT>(v_cache, v_scale, slot, kvh, sub);
            } else {
                *reinterpret_cast<uint4*>(&smem.kv.s_k[t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem.kv.s_v[t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        int w_len = min(16, max(0, tile_len - t_base));
        if (w_len > 0) {
            // 2. QK^T via 8x ldmatrix.x4 + 16x m16n8k16 MMAs
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                uint32_t kb0, kb1, kb2, kb3;
                ldmatrix_x4(kb0, kb1, kb2, kb3, &smem.kv.s_k[k_ld_row][kt * 16 + k_ld_col]);
                mma_m16n8k16_f32_f16(s0, s1, s2, s3,
                                     q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                     kb0, kb1, s0, s1, s2, s3);
                mma_m16n8k16_f32_f16(s4, s5, s6, s7,
                                     q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                     kb2, kb3, s4, s5, s6, s7);
            }

            if (w_len < 16) {
                if (c0 >= w_len) { s0 = -1e30f; s2 = -1e30f; }
                if (c1 >= w_len) { s1 = -1e30f; s3 = -1e30f; }
                if (c4 >= w_len) { s4 = -1e30f; s6 = -1e30f; }
                if (c5 >= w_len) { s5 = -1e30f; s7 = -1e30f; }
            }

            float rmax0 = fmaxf(fmaxf(s0, s1), fmaxf(s4, s5));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
            float rmax1 = fmaxf(fmaxf(s2, s3), fmaxf(s6, s7));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

            float m0_new = fmaxf(m0, rmax0);
            float m1_new = fmaxf(m1, rmax1);
            float alpha0 = (m0 > -1e29f) ? __expf(m0 - m0_new) : 0.0f;
            float alpha1 = (m1 > -1e29f) ? __expf(m1 - m1_new) : 0.0f;

            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                o[nt][0] *= alpha0;
                o[nt][1] *= alpha0;
                o[nt][2] *= alpha1;
                o[nt][3] *= alpha1;
            }

            float ps0 = (c0 < w_len) ? __expf(s0 - m0_new) : 0.0f;
            float ps1 = (c1 < w_len) ? __expf(s1 - m0_new) : 0.0f;
            float ps4 = (c4 < w_len) ? __expf(s4 - m0_new) : 0.0f;
            float ps5 = (c5 < w_len) ? __expf(s5 - m0_new) : 0.0f;
            float ps2 = (c0 < w_len) ? __expf(s2 - m1_new) : 0.0f;
            float ps3 = (c1 < w_len) ? __expf(s3 - m1_new) : 0.0f;
            float ps6 = (c4 < w_len) ? __expf(s6 - m1_new) : 0.0f;
            float ps7 = (c5 < w_len) ? __expf(s7 - m1_new) : 0.0f;

            float rsum0 = ps0 + ps1 + ps4 + ps5;
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
            float rsum1 = ps2 + ps3 + ps6 + ps7;
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);
            d0 = d0 * alpha0 + rsum0; m0 = m0_new;
            d1 = d1 * alpha1 + rsum1; m1 = m1_new;

            __half2 hp0 = __floats2half2_rn(ps0, ps1);
            __half2 hp1 = __floats2half2_rn(ps2, ps3);
            __half2 hp2 = __floats2half2_rn(ps4, ps5);
            __half2 hp3 = __floats2half2_rn(ps6, ps7);
            uint32_t pa0 = *reinterpret_cast<uint32_t*>(&hp0);
            uint32_t pa1 = *reinterpret_cast<uint32_t*>(&hp1);
            uint32_t pa2 = *reinterpret_cast<uint32_t*>(&hp2);
            uint32_t pa3 = *reinterpret_cast<uint32_t*>(&hp3);

            // 3. P @ V via 16x hardware ldmatrix.x2.trans + 16x m16n8k16 MMAs!
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                uint32_t vb0, vb1;
                ldmatrix_x2_trans(vb0, vb1, &smem.kv.s_v[v_ld_row][nt * 8]);
                mma_m16n8k16_f32_f16(o[nt][0], o[nt][1], o[nt][2], o[nt][3],
                                     pa0, pa1, pa2, pa3,
                                     vb0, vb1,
                                     o[nt][0], o[nt][1], o[nt][2], o[nt][3]);
            }
        }
        __syncthreads();
    }

    // 4. Reduce the 2 token-slices per M-tile (all 16 queries at once in 16.4 KB SMEM!)
    for (int mt = 0; mt < NM; ++mt) {
        if (m_tile == mt) {
            if (tid4 == 0) {
                smem.red.m[grp][slice]     = m0;
                smem.red.d[grp][slice]     = d0;
                smem.red.m[grp + 8][slice] = m1;
                smem.red.d[grp + 8][slice] = d1;
            }
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                smem.red.o[slice][grp][nt * 8 + c0]     = o[nt][0];
                smem.red.o[slice][grp][nt * 8 + c1]     = o[nt][1];
                smem.red.o[slice][grp + 8][nt * 8 + c0] = o[nt][2];
                smem.red.o[slice][grp + 8][nt * 8 + c1] = o[nt][3];
            }
        }
        __syncthreads();

        for (int i = tid; i < 16 * HEAD_DIM; i += THREADS) {
            int qi = i >> 7;
            int dim = i & 127;
            int q_glob = mt * 16 + qi;
            if (q_glob < Q_CNT) {
                int gqa = q_glob % GQA_RATIO;
                int b = q_glob / GQA_RATIO;
                int h = kvh * GQA_RATIO + gqa;
                int idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
                float gm = fmaxf(smem.red.m[qi][0], smem.red.m[qi][1]);
                float gd = 0.0f, go = 0.0f;
                if (smem.red.d[qi][0] > 0.0f) {
                    float sc = __expf(smem.red.m[qi][0] - gm);
                    gd += smem.red.d[qi][0] * sc;
                    go += smem.red.o[0][qi][dim] * sc;
                }
                if (smem.red.d[qi][1] > 0.0f) {
                    float sc = __expf(smem.red.m[qi][1] - gm);
                    gd += smem.red.d[qi][1] * sc;
                    go += smem.red.o[1][qi][dim] * sc;
                }
                if (dim == 0) { s_m_out[idx] = gm; s_d_out[idx] = gd; }
                s_o_out[(size_t)idx * HEAD_DIM + dim] = go;
            }
        }
        if (mt + 1 < NM) __syncthreads();
    }
}

// Stateless SplitMix64 uniform (0, 1) and full-64-bit Gumbel(0, 1) noise helpers for CUDA-graph sampling.
// 64-bit floating-point Gumbel technique adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0):
// keeps 24 significant bits down to 2^-64 so far-tail tokens (down to ~e^-44) sample accurately without Inf/NaN.
__device__ __forceinline__ uint64_t hash_u64(uint64_t seed, int32_t pos, int32_t tok) {
    uint64_t h = seed ^ ((uint64_t)(uint32_t)pos * 0x9E3779B97F4A7C15ULL)
                      ^ ((uint64_t)(uint32_t)tok * 0xBF58476D1CE4E5B9ULL);
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

__device__ __forceinline__ float hash_u01(uint64_t seed, int32_t pos, int32_t tok) {
    uint64_t h = hash_u64(seed, pos, tok);
    return ((float)(h >> 41) + 0.5f) * (1.0f / 8388608.0f);
}

__device__ __forceinline__ float hash_gumbel(uint64_t seed, int32_t pos, int32_t tok) {
    uint64_t k = hash_u64(seed, pos, tok);
    int z = min(__clzll(static_cast<long long>(k)), 40);
    float m = static_cast<float>(static_cast<uint32_t>(k >> (40 - z))) + (z == 40 ? 0.5f : 0.0f);
    float v = m * __int_as_float((127 - 24 - z) << 23); // strictly in (0, 1)
    float e = (v < 0.5f) ? -log1pf(-v) : -logf(1.0f - v);
    return -logf(e);
}

// Blackwell / Hopper Thread-Block Cluster + DSMEM mbarrier primitives (CUDA 13.1+ sm_120a safe).
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine (Apache-2.0).
__device__ __forceinline__ uint32_t cluster_smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t cluster_remote_u32(uint32_t addr, int dst_rank) {
    uint32_t r;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(dst_rank));
    return r;
}

__device__ __forceinline__ void cluster_mb_init(uint32_t mb) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
}

__device__ __forceinline__ void cluster_fence_mb_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void cluster_mb_expect_tx(uint32_t mb, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cluster_mb_wait(uint32_t mb, uint32_t parity = 0u) {
    uint32_t done = 0u;
    while (!done) {
        asm volatile("{\n.reg .pred p;\n"
                     "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(mb), "r"(parity) : "memory");
    }
}

__device__ __forceinline__ void cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait_acquire() {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_st_async_f4(uint32_t dst_addr, float4 v, uint32_t remote_mb) {
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
                 :: "r"(dst_addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(remote_mb) : "memory");
}

__device__ __forceinline__ void cp_async16_cg(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(cluster_smem_u32(smem_dst)), "l"(gmem_src) : "memory");
}

template <class... KArgs, class... Args>
inline void launch_pdl_cluster(
    void (*kernel)(KArgs...), dim3 grid, int block, int smem_bytes, int cluster_x,
    cudaStream_t stream, Args&&... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block);
    cfg.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int n_attr = 0;
    attr[n_attr].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n_attr].val.programmaticStreamSerializationAllowed = 1;
    ++n_attr;
    if (cluster_x > 1) {
        attr[n_attr].id = cudaLaunchAttributeClusterDimension;
        attr[n_attr].val.clusterDim.x = static_cast<unsigned int>(cluster_x);
        attr[n_attr].val.clusterDim.y = 1u;
        attr[n_attr].val.clusterDim.z = 1u;
        ++n_attr;
    }
    cfg.attrs = attr;
    cfg.numAttrs = n_attr;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

// Whether `kernel` can run as `cs`-CTA clusters on this GPU. A cluster must fit in one GPC: GB202
// (RTX 5090) has 16 SMs per GPC, the RTX 5080 Laptop's largest cluster is 10 CTAs, and a launch
// past the limit fails with cudaErrorInvalidClusterSize.
template <class Kernel>
inline bool cluster_size_fits(Kernel kernel, int cs, int grid_x, int grid_y, int block, size_t smem) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid_x, grid_y);
    cfg.blockDim = dim3(block);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = static_cast<unsigned int>(cs);
    attr.val.clusterDim.y = 1u;
    attr.val.clusterDim.z = 1u;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int clusters = 0;
    const cudaError_t err = cudaOccupancyMaxActiveClusters(&clusters, (const void*)kernel, &cfg);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return clusters > 0;
}

} // namespace squid

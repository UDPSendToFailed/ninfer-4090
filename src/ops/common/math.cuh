#pragma once

#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float log2_approx(float x) {
    float y;
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float silu(float x) {
    constexpr float kNegLog2e = -1.4426950408889634f;
    const float denom = 1.0f + exp2_approx(x * kNegLog2e);
    return x * __frcp_rn(denom);
}

__device__ __forceinline__ float sigmoid(float x) {
    constexpr float kNegLog2e = -1.4426950408889634f;
    const float denom = 1.0f + exp2_approx(x * kNegLog2e);
    return __frcp_rn(denom);
}

__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp2_approx(x * 1.4426950408889634f);
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2   = 0.6931471805599453f;
    return kLn2 * log2_approx(1.0f + exp2_approx(x * kLog2e));
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    std::uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
}

__device__ __forceinline__ __nv_bfloat162 float2_to_bf16x2(float2 value) {
    return __float22bfloat162_rn(value);
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t bits) {
    __nv_bfloat162 val;
    std::memcpy(&val, &bits, sizeof(val));
    return bf16x2_to_float2(val);
}

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    __half2 val;
    std::memcpy(&val, &bits, sizeof(val));
    return val;
}

} // namespace ninfer::ops

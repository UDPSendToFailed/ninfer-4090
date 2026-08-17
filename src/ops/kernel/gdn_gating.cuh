#pragma once

// ninfer::ops - gdn_gating kernel: elementwise GDN gate prep over [48,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

__global__ void gdn_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  const float* A_log, const float* dt_bias, float* g, float* beta,
                                  std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    const std::int64_t n_vec4 = n / 4;
    const auto* a_vec = reinterpret_cast<const __nv_bfloat162*>(a);
    const auto* b_vec = reinterpret_cast<const __nv_bfloat162*>(b);
    auto* g_vec       = reinterpret_cast<float4*>(g);
    auto* beta_vec    = reinterpret_cast<float4*>(beta);

    for (std::int64_t i = start; i < n_vec4; i += stride) {
        // Fast head indexing: (i * 4) % 48 == (i % 12) * 4
        const int base_h = static_cast<int>((static_cast<std::uint32_t>(i) % 12U) * 4U);
        const int h0 = base_h + 0;
        const int h1 = base_h + 1;
        const int h2 = base_h + 2;
        const int h3 = base_h + 3;

        // Vectorized 128-bit loads (8 bytes per pair)
        const __nv_bfloat162 a_pair0 = a_vec[i * 2 + 0];
        const __nv_bfloat162 a_pair1 = a_vec[i * 2 + 1];
        const __nv_bfloat162 b_pair0 = b_vec[i * 2 + 0];
        const __nv_bfloat162 b_pair1 = b_vec[i * 2 + 1];

        // Unpack & Fast Hardware Math in registers
        float4 g_out, beta_out;

        // Lane 0
        const float av0 = __low2float(a_pair0), bv0 = __low2float(b_pair0);
        g_out.x = -__expf(A_log[h0]) * softplus(av0 + dt_bias[h0]);
        beta_out.x = sigmoid(bv0);

        // Lane 1
        const float av1 = __high2float(a_pair0), bv1 = __high2float(b_pair0);
        g_out.y = -__expf(A_log[h1]) * softplus(av1 + dt_bias[h1]);
        beta_out.y = sigmoid(bv1);

        // Lane 2
        const float av2 = __low2float(a_pair1), bv2 = __low2float(b_pair1);
        g_out.z = -__expf(A_log[h2]) * softplus(av2 + dt_bias[h2]);
        beta_out.z = sigmoid(bv2);

        // Lane 3
        const float av3 = __high2float(a_pair1), bv3 = __high2float(b_pair1);
        g_out.w = -__expf(A_log[h3]) * softplus(av3 + dt_bias[h3]);
        beta_out.w = sigmoid(bv3);

        // Vectorized 128-bit stores
        g_vec[i]    = g_out;
        beta_vec[i] = beta_out;
    }

    // Scalar tail handling for boundary tokens
    for (std::int64_t i = n_vec4 * 4 + start; i < n; i += stride) {
        const int h    = static_cast<int>(i % 48);
        const float av = __bfloat162float(a[i]);
        const float bv = __bfloat162float(b[i]);
        const float sp = softplus(av + dt_bias[h]);
        g[i]           = -__expf(A_log[h]) * sp;
        beta[i]        = sigmoid(bv);
    }
}

} // namespace ninfer::ops

#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <class T, class = void>
struct is_q4_tile_epilogue : std::false_type {};

template <class T>
struct is_q4_tile_epilogue<T, std::void_t<decltype(T::kIsTileEpilogue)>>
    : std::bool_constant<T::kIsTileEpilogue> {};

template <class T>
inline constexpr bool is_q4_tile_epilogue_v = is_q4_tile_epilogue<T>::value;

struct Q4SmallTMmaStoreEpilogue {};

struct Q4SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int OutputRows, int InputRows>
struct Q4SmallTGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

template <int InputRows>
using Q4DraftHeadGeometry = Q4SmallTGeometry<131072, InputRows>;

template <int SplitRow>
struct Q4SmallTSplitEpilogue {
    __nv_bfloat16* out;
    __nv_bfloat16* out_tail;
    std::int32_t out_ld;
    std::int32_t tail_ld;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            if (row < SplitRow) {
                out[static_cast<std::int64_t>(col0) * out_ld + row] =
                    __float2bfloat16_rn(projected.x);
            } else {
                out_tail[static_cast<std::int64_t>(col0) * tail_ld + (row - SplitRow)] =
                    __float2bfloat16_rn(projected.x);
            }
            if (row + 8 < SplitRow) {
                out[static_cast<std::int64_t>(col0) * out_ld + row + 8] =
                    __float2bfloat16_rn(projected.z);
            } else {
                out_tail[static_cast<std::int64_t>(col0) * tail_ld + (row + 8 - SplitRow)] =
                    __float2bfloat16_rn(projected.z);
            }
        }
        if (col0 + 1 < ActiveCols) {
            if (row < SplitRow) {
                out[static_cast<std::int64_t>(col0 + 1) * out_ld + row] =
                    __float2bfloat16_rn(projected.y);
            } else {
                out_tail[static_cast<std::int64_t>(col0 + 1) * tail_ld + (row - SplitRow)] =
                    __float2bfloat16_rn(projected.y);
            }
            if (row + 8 < SplitRow) {
                out[static_cast<std::int64_t>(col0 + 1) * out_ld + row + 8] =
                    __float2bfloat16_rn(projected.w);
            } else {
                out_tail[static_cast<std::int64_t>(col0 + 1) * tail_ld + (row + 8 - SplitRow)] =
                    __float2bfloat16_rn(projected.w);
            }
        }
    }
};

struct Q4SmallTStrideEpilogue {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * out_ld + row]     = __float2bfloat16_rn(projected.x);
            out[static_cast<std::int64_t>(col0) * out_ld + row + 8] = __float2bfloat16_rn(projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * out_ld + row]     = __float2bfloat16_rn(projected.y);
            out[static_cast<std::int64_t>(col0 + 1) * out_ld + row + 8] = __float2bfloat16_rn(projected.w);
        }
    }
};

struct Q4DraftSmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

__device__ __forceinline__ int q4_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q4SmallTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q4_small_t_bf16_pair(std::uint8_t packed) {
    const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
    const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
    Q4SmallTBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4SmallTMmaStoreEpilogue,
          class RowPolicy = Q4SmallTMmaIdentityRows>
__launch_bounds__(256, 2) __global__
    void q4_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                               RowPolicy row_policy = {}) {
    using Schedule              = Q4DraftSmallTSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    // To strictly remain within the 48 KB static shared memory limit on sm_89,
    // double-buffer (ping-pong) for TileCols <= 16 (<= 41.5 KB), and single-buffer for TileCols > 16.
    constexpr int kStages       = (TileCols <= 16) ? 2 : 1;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);

    union SharedStorage {
        struct {
            std::uint8_t codes[kStages][kRowsPerCta][kGroupK / 2];
            __nv_bfloat16 activations[kStages][kWarps][kTileCols * kTileK];
            std::uint16_t scales[kStages][kRowsPerCta][kWarps];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int row0    = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    const auto stage_x = [&](int stage, int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[stage][warp][col * kTileK + q4_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16>(
                dst,
                &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK + k8 * 8]);
        }
    };

    const auto stage_weight = [&](int stage, int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(row0, row);
            for (int chunk = lane; chunk < kGroupK / 32; chunk += 32) {
                cp_async<16, Schedule::kCodeCache>(
                    &code_shared[stage][row][chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                        chunk * 16);
            }
        }
        for (int row = tid; row < kRowsPerCta; row += kWarps * 32) {
            const int weight_row = row_policy.weight_row(row0, row);
            cp_async<16>(&scale_shared[stage][row][0],
                         scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                                   group_k0 / 64) *
                                       2);
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kNt][4]   = {};

    if constexpr (kStages == 2) {
        // 1. Prologue: Asynchronously prefetch stage 0.
        stage_weight(0, 0);
        stage_x(0, 0);
        cp_commit();

        int curr_stage = 0;

        // 2. Main Pipelined Loop: overlap stage s+1 async DMA with stage s MMA compute.
#pragma unroll 1
        for (int group_index = 0; group_index < kGroups; ++group_index) {
            const int next_stage = 1 - curr_stage;
            const int next_group = group_index + 1;

            if (next_group < kGroups) {
                stage_weight(next_stage, next_group * kGroupK);
                stage_x(next_stage, next_group * kGroupK);
                cp_commit();
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();

            float group_acc[kNt][4] = {};

#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int byte_col = warp_koff / 2 + ks * 8 + lid;
                const unsigned af0 = q4_small_t_bf16_pair(code_shared[curr_stage][gid][byte_col]);
                const unsigned af1 = q4_small_t_bf16_pair(code_shared[curr_stage][gid + 8][byte_col]);
                const unsigned af2 = q4_small_t_bf16_pair(code_shared[curr_stage][gid][byte_col + 4]);
                const unsigned af3 = q4_small_t_bf16_pair(code_shared[curr_stage][gid + 8][byte_col + 4]);
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    unsigned bf0, bf1;
                    const int br = nt * 8 + b_rin;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&x_shared[curr_stage][k_split][br * kTileK + q4_small_t_swizzle_64(
                                                                               br, ks * 16 + b_koff)]));
                    mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                             af0, af1, af2, af3, bf0, bf1);
                }
            }

            const float top_scale = __half2float(__ushort_as_half(scale_shared[curr_stage][gid][k_split]));
            const float bot_scale = __half2float(__ushort_as_half(scale_shared[curr_stage][gid + 8][k_split]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
                acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
                acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
                acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
            }

            __syncthreads();
            curr_stage = next_stage;
        }
    } else {
        // Fallback for kStages == 1 (TileCols > 16)
        stage_weight(0, 0);
        stage_x(0, 0);
        cp_commit();
        cp_wait<0>();
        __syncthreads();

#pragma unroll
        for (int group_index = 0; group_index < kGroups; ++group_index) {
            const int group_k0      = group_index * kGroupK;
            float group_acc[kNt][4] = {};

#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int byte_col = warp_koff / 2 + ks * 8 + lid;
                const unsigned af0 = q4_small_t_bf16_pair(code_shared[0][gid][byte_col]);
                const unsigned af1 = q4_small_t_bf16_pair(code_shared[0][gid + 8][byte_col]);
                const unsigned af2 = q4_small_t_bf16_pair(code_shared[0][gid][byte_col + 4]);
                const unsigned af3 = q4_small_t_bf16_pair(code_shared[0][gid + 8][byte_col + 4]);
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    unsigned bf0, bf1;
                    const int br = nt * 8 + b_rin;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&x_shared[0][k_split][br * kTileK + q4_small_t_swizzle_64(
                                                                               br, ks * 16 + b_koff)]));
                    mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                             af0, af1, af2, af3, bf0, bf1);
                }
            }

            const float top_scale = __half2float(__ushort_as_half(scale_shared[0][gid][k_split]));
            const float bot_scale = __half2float(__ushort_as_half(scale_shared[0][gid + 8][k_split]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
                acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
                acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
                acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
            }

            if (group_index + 1 < kGroups) {
                __syncthreads();
                stage_weight(0, group_k0 + kGroupK);
                stage_x(0, group_k0 + kGroupK);
                cp_commit();
                cp_wait<0>();
                __syncthreads();
            }
        }
    }

    // Inter-warp reduction across 8 warps in shared memory.
    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 partner =
                load_vec<float4>(partial + (((k_split + 1) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                          make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + nt) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            acc[nt][0] = sum.x;
            acc[nt][1] = sum.y;
            acc[nt][2] = sum.z;
            acc[nt][3] = sum.w;
        }

        if constexpr (is_q4_tile_epilogue_v<Epilogue>) {
            epilogue.template store_tile<ActiveCols, TileCols, Geometry::kOutputRows>(partial, row0, gid, lane, acc);
        } else if constexpr (std::is_same_v<Epilogue, Q4SmallTMmaStoreEpilogue>) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                if (col0 < ActiveCols) {
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid] =
                        __float2bfloat16_rn(acc[nt][0]);
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid + 8] =
                        __float2bfloat16_rn(acc[nt][2]);
                }
                if (col0 + 1 < ActiveCols) {
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 + gid] =
                        __float2bfloat16_rn(acc[nt][1]);
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 + gid +
                        8] = __float2bfloat16_rn(acc[nt][3]);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                epilogue.template store<ActiveCols>(row0 + gid, col0,
                    make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
}

} // namespace ninfer::ops::detail

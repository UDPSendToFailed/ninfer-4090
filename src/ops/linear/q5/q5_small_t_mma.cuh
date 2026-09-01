#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <class T, class = void>
struct is_q5_tile_epilogue : std::false_type {};

template <class T>
struct is_q5_tile_epilogue<T, std::void_t<decltype(T::kIsTileEpilogue)>>
    : std::bool_constant<T::kIsTileEpilogue> {};

template <class T>
inline constexpr bool is_q5_tile_epilogue_v = is_q5_tile_epilogue<T>::value;

struct Q5SmallTMmaStoreEpilogue {
    __device__ __forceinline__ void store(__nv_bfloat16* out, int out_ld, int row, int col,
                                          float acc) const {
        out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16_rn(acc);
    }
};

struct Q5SmallTMmaResidualEpilogue {
    __device__ __forceinline__ void store(__nv_bfloat16* out, int out_ld, int row, int col,
                                          float acc) const {
        auto* dst = &out[static_cast<std::int64_t>(col) * out_ld + row];
        *dst      = __hadd(*dst, __float2bfloat16_rn(acc));
    }
};

template <int SplitRow>
struct Q5SmallTSplitEpilogue {
    __nv_bfloat16* out_tail;
    std::int32_t tail_ld;

    __device__ __forceinline__ void store(__nv_bfloat16* out, int out_ld, int row, int col,
                                          float acc) const {
        if (row < SplitRow) {
            out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16_rn(acc);
        } else {
            out_tail[static_cast<std::int64_t>(col) * tail_ld + (row - SplitRow)] =
                __float2bfloat16_rn(acc);
        }
    }
};

struct Q5SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int OutputRows, int InputRows>
struct Q5SmallTGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / Q5RowSplitStorage::kGroupK;
};

struct Q5SmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 2;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp; // 512
    static constexpr int kRowsPerCta        = 16;
};

__device__ __forceinline__ int q5_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q5SmallTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q5_small_t_bf16_pair(std::uint8_t code_byte,
                                                        std::uint8_t high_byte,
                                                        int shift) {
    const int q0 = ((static_cast<int>(code_byte & 0x0fu) |
                     (((static_cast<int>(high_byte) >> shift) & 1) << 4)) ^ 0x10) - 0x10;
    const int q1 = ((static_cast<int>(code_byte >> 4) |
                     (((static_cast<int>(high_byte) >> (shift + 1)) & 1) << 4)) ^ 0x10) - 0x10;
    Q5SmallTBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q5SmallTMmaStoreEpilogue,
          class RowPolicy = Q5SmallTMmaIdentityRows>
__launch_bounds__(256, 2) __global__
    void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ high,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out,
                               std::int32_t in_ld,
                               std::int32_t out_ld,
                               Epilogue epilogue = {},
                               RowPolicy row_policy = {}) {
    using Schedule              = Q5SmallTSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kOutputRows   = Geometry::kOutputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp; // 64
    constexpr int kWarps        = Schedule::kKWarps;       // 8
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;   // 16
    constexpr int kGroupK       = Schedule::kGroupK;       // 512
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);

    // Double-buffered (ping-pong) shared storage.
    union SharedStorage {
        struct {
            std::uint8_t codes[2][kRowsPerCta][kGroupK / 2];           // 2 x 16 x 256 B = 8 KB
            std::uint8_t high[2][kRowsPerCta][kGroupK / 8];            // 2 x 16 x 64 B  = 2 KB
            std::uint16_t scales[2][kRowsPerCta][kWarps];              // 2 x 16 x 8 x 2 B = 512 B
            __nv_bfloat16 activations[2][kWarps][kTileCols * kTileK];  // 2 x 8 x (Cols * 64) x 2 B
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& high_shared  = shared.staging.high;
    auto& scale_shared = shared.staging.scales;
    auto& x_shared     = shared.staging.activations;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int row0    = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    // Fully coalesced activation loader: each warp stages its own 64-element K slice.
    const auto stage_x = [&](int stage, int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[stage][warp][col * kTileK + q5_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16>(
                dst,
                &x[static_cast<std::int64_t>(col) * in_ld + group_k0 + warp * kTileK + k8 * 8]);
        }
    };

    // Fully coalesced weight loader: each warp loads 2 complete rows along the K dimension.
    // Threads 0..15 load row 0 (256 B codes, 64 B high, 16 B scale) via 16-byte contiguous vector loads.
    // Threads 16..31 load row 1 via 16-byte contiguous vector loads.
    // All 32 threads in the warp execute coalesced 128-byte cache line transactions.
    const auto stage_weight = [&](int stage, int group_k0) {
        const int quant_group0 = group_k0 / Q5RowSplitStorage::kGroupK;
        const int local_row0   = warp * 2;
        const int local_row1   = local_row0 + 1;
        const int weight_row0  = row_policy.weight_row(row0, local_row0);
        const int weight_row1  = row_policy.weight_row(row0, local_row1);

        const std::int64_t global_grp0 =
            static_cast<std::int64_t>(weight_row0) * Geometry::kGroupsPerRow + quant_group0;
        const std::int64_t global_grp1 =
            static_cast<std::int64_t>(weight_row1) * Geometry::kGroupsPerRow + quant_group0;

        // 1. Codes: 256 bytes per row (16 chunks of 16 bytes).
        if (lane < 16) {
            cp_async<16, Schedule::kCodeCache>(
                &code_shared[stage][local_row0][lane * 16],
                codes + global_grp0 * Q5RowSplitStorage::kCodeBytesPerGroup + lane * 16);
        } else {
            const int chunk = lane - 16;
            cp_async<16, Schedule::kCodeCache>(
                &code_shared[stage][local_row1][chunk * 16],
                codes + global_grp1 * Q5RowSplitStorage::kCodeBytesPerGroup + chunk * 16);
        }

        // 2. High bits: 64 bytes per row (4 chunks of 16 bytes).
        if (lane < 4) {
            cp_async<16, Schedule::kCodeCache>(
                &high_shared[stage][local_row0][lane * 16],
                high + global_grp0 * Q5RowSplitStorage::kHighBytesPerGroup + lane * 16);
        } else if (lane < 8) {
            const int chunk = lane - 4;
            cp_async<16, Schedule::kCodeCache>(
                &high_shared[stage][local_row1][chunk * 16],
                high + global_grp1 * Q5RowSplitStorage::kHighBytesPerGroup + chunk * 16);
        }

        // 3. Scales: 16 bytes per row (1 chunk of 16 bytes).
        if (lane == 0) {
            cp_async<16>(
                &scale_shared[stage][local_row0][0],
                scales + global_grp0 * Q5RowSplitStorage::kScaleBytesPerGroup);
        } else if (lane == 1) {
            cp_async<16>(
                &scale_shared[stage][local_row1][0],
                scales + global_grp1 * Q5RowSplitStorage::kScaleBytesPerGroup);
        }
    };

    const int b_rin  = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    float acc[kNt][4] = {};

    // 1. Prologue: Prefetch Stage 0.
    stage_weight(0, 0);
    stage_x(0, 0);
    cp_commit();

    int curr_stage = 0;

    // 2. Pipelined Loop: overlap Stage s+1 async DMA with Stage s MMA compute.
#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int next_stage = 1 - curr_stage;
        const int next_group = group_index + 1;

        // Asynchronously issue next stage while current stage is in flight.
        if (next_group < kGroups) {
            stage_weight(next_stage, next_group * kGroupK);
            stage_x(next_stage, next_group * kGroupK);
            cp_commit();
            cp_wait<1>(); // Wait for curr_stage, keeping next_stage streaming
        } else {
            cp_wait<0>(); // Final stage: drain all in-flight copies
        }
        __syncthreads();

        float group_acc[kNt][4] = {};

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte0 = k_split * 32 + ks * 8 + lid;
            const int byte1 = byte0 + 4;
            const int shift = lid * 2;

            const std::uint8_t h0_top = high_shared[curr_stage][gid][k_split * 8 + ks * 2];
            const std::uint8_t h1_top = high_shared[curr_stage][gid][k_split * 8 + ks * 2 + 1];
            const std::uint8_t h0_bot = high_shared[curr_stage][gid + 8][k_split * 8 + ks * 2];
            const std::uint8_t h1_bot = high_shared[curr_stage][gid + 8][k_split * 8 + ks * 2 + 1];

            const unsigned af0 = q5_small_t_bf16_pair(code_shared[curr_stage][gid][byte0], h0_top, shift);
            const unsigned af1 = q5_small_t_bf16_pair(code_shared[curr_stage][gid + 8][byte0], h0_bot, shift);
            const unsigned af2 = q5_small_t_bf16_pair(code_shared[curr_stage][gid][byte1], h1_top, shift);
            const unsigned af3 = q5_small_t_bf16_pair(code_shared[curr_stage][gid + 8][byte1], h1_bot, shift);

#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned bf0, bf1;
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&x_shared[curr_stage][k_split][br * kTileK + q5_small_t_swizzle_64(
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
            const float4 other =
                load_vec<float4>(partial + (((k_split + 1) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += other.x;
            acc[nt][1] += other.y;
            acc[nt][2] += other.z;
            acc[nt][3] += other.w;
        }
    }
    __syncthreads();

    if (k_split == 2 || k_split == 6) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + (((k_split / 2) * kNt + nt) * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();

    if (k_split == 0 || k_split == 4) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 other =
                load_vec<float4>(partial + ((((k_split + 2) / 2) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += other.x;
            acc[nt][1] += other.y;
            acc[nt][2] += other.z;
            acc[nt][3] += other.w;
        }
    }
    __syncthreads();

    if (k_split == 4) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + (nt * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 other = load_vec<float4>(partial + (nt * 32 + lane) * 4);
            acc[nt][0] += other.x;
            acc[nt][1] += other.y;
            acc[nt][2] += other.z;
            acc[nt][3] += other.w;
        }

        if constexpr (is_q5_tile_epilogue_v<Epilogue>) {
            epilogue.template store_tile<ActiveCols, TileCols, kOutputRows>(partial, row0, gid, lane, acc);
        } else {
            const int out_row0 = row_policy.weight_row(row0, gid);
            const int out_row1 = row_policy.weight_row(row0, gid + 8);
            const int col_base = (lane & 3) * 2;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col0 = nt * 8 + col_base;
                const int col1 = col0 + 1;
                if (col0 < ActiveCols && out_row0 < kOutputRows) {
                    epilogue.store(out, out_ld, out_row0, col0, acc[nt][0]);
                }
                if (col0 < ActiveCols && out_row1 < kOutputRows) {
                    epilogue.store(out, out_ld, out_row1, col0, acc[nt][2]);
                }
                if (col1 < ActiveCols && out_row0 < kOutputRows) {
                    epilogue.store(out, out_ld, out_row0, col1, acc[nt][1]);
                }
                if (col1 < ActiveCols && out_row1 < kOutputRows) {
                    epilogue.store(out, out_ld, out_row1, col1, acc[nt][3]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

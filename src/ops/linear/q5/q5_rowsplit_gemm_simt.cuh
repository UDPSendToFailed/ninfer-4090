#pragma once

// Warp-per-row small-T row-split low-bit GEMM: out[N,T] = W[N,K] . x[K,T].
//
// This is Q5's format-local small-column path. The design target is the DRAM
// roofline: stream the Q5 payload once per column tile at copy-ceiling
// bandwidth, so the cost of T<=tile width is nearly flat versus T==1.
//
//   - One warp owns one output row. blockIdx.y selects a tile of kTt activation
//     columns; for T <= kTt the weights are streamed exactly once.
//   - K is processed in 1024-value slabs staged through shared memory with a
//     cp.async double buffer (same structure as the tuned T==1 Q5 core): all
//     weight-plane reads are coalesced 128-bit loads, so DRAM latency is hidden
//     without relying on occupancy alone. Deeper pipelines and launch-bounds
//     occupancy caps were both measured slower (spills / no gain, see report).
//   - Consume is phase-interleaved: in phase c (0..3), lane L owns the 8
//     consecutive K-values [256c + 8L, 256c + 8L + 8). The warp's 32 x loads per
//     (column, phase) are one 16-byte uint4 per lane covering 512 *consecutive*
//     bytes, i.e. minimal L1 wavefronts (a per-lane-contiguous layout would
//     stride lanes 64B apart and burn 4x the L1 throughput on the same bytes).
//     The weight planes index out conflict-free or broadcast under the same
//     ownership.
//   - Dequant uses the fp16 mantissa trick: nibbles are OR-ed into the mantissa
//     of 1024.0f16 (plus Q5 high bits at mantissa bit 4), the sign
//     fixup is a constant xor folded into the packed word, and one hsub2 yields
//     two exact signed values; the group scale is applied per 8-value chunk.
//     fp32 FMA into per-column accumulators; warp-shuffle reduction per column.
//   - kTt is 4 or 8 only. Larger column tiles blow past the register budget
//     (kTt=16 needs ~98 regs -> 2 blocks/SM -> latency-bound at ~20% of DRAM),
//     so T > 8 re-streams the weights once per 8-column tile instead; the mma
//     tensor-core GEMM takes over where that stops winning.
//
// Correctness for arbitrary shapes: full 1024-value slabs require k % 8 == 0
// (16-byte aligned x columns) and cover [0, k/1024*1024); every remaining group
// (tail of odd k, or all groups when k % 8 != 0) uses the scalar per-pair path
// reading global memory directly, masked at the k boundary. Weights in the
// padded region [k, padded_k) are never used.

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

// Q5 slab traits + dequant. A slab is 1024 K-values:
//   nibble/code bytes : 32 B per group  -> kNibU4 uint4 per slab
//   high-plane bytes  : 0/8/16 B per group -> kHighU4 uint4 per slab
//   scales            : 2 B per group -> kScaleU32 4-byte words per slab
// (Scales are staged with 4-byte cp.async because a row's scale plane is only
// guaranteed 4-byte aligned for generic kg.)
//
// dequant_chunk(s_nib, s_hi, s_sc, c, lane, w): dequantize the lane's 8 values
// of phase c into w[0..7], scale applied. All shared reads are conflict-free or
// broadcast under the phase-interleaved ownership.

struct Q5RowSplitSimtSchedule {
    using Codec                             = Q5ScalarDecodeAtom;
    static constexpr int kNibU4             = 32;
    static constexpr int kHighU4            = 8;
    static constexpr int kScaleU32          = 8;
    static constexpr int kHighBytesPerGroup = 8;

    // u = lo | hi<<4, signed s = (u ^ 16) - 16 = lo + 16*(hi^1) - 16. Build
    // 1024 + lo + 16*(hi^1) in fp16 (hi^1 at mantissa bit 4), subtract 1040.
    __device__ static __forceinline__ void dequant_chunk(const uint4* s_nib, const uint4* s_hi,
                                                         const std::uint32_t* s_sc, int c, int lane,
                                                         float (&w)[8]) {
        const std::uint32_t word = reinterpret_cast<const std::uint32_t*>(s_nib)[c * 32 + lane];
        const std::uint32_t hc = reinterpret_cast<const std::uint8_t*>(s_hi)[c * 32 + lane] ^ 0xffu;
        const float scale      = __half2float(
            __ushort_as_half(reinterpret_cast<const std::uint16_t*>(s_sc)[c * 4 + (lane >> 3)]));
        const __half2 bias = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(half2_from_bits(bits), bias);
            const float2 f  = __half22float2(h);
            w[p]            = f.x * scale;
            w[p + 4]        = f.y * scale;
        }
    }
};

template <class SC>
__device__ __forceinline__ void
q5_simt_issue_slab(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi,
                   std::uint32_t* __restrict__ s_sc, const std::uint8_t* __restrict__ code_row,
                   const std::uint8_t* __restrict__ high_row,
                   const std::uint8_t* __restrict__ scale_row, int slab, int lane) {
    static_assert(SC::kNibU4 % 32 == 0, "nibble plane must be a whole number of warp copies");
#pragma unroll
    for (int j = 0; j < SC::kNibU4 / 32; ++j) {
        const int i = j * 32 + lane;
        pipe_copy<16>(&s_nib[i],
                      code_row + static_cast<std::int64_t>(slab) * (SC::kNibU4 * 16) + i * 16);
    }
    if constexpr (SC::kHighU4 > 0) {
        if (lane < SC::kHighU4) {
            pipe_copy<16>(&s_hi[lane], high_row +
                                           static_cast<std::int64_t>(slab) * (SC::kHighU4 * 16) +
                                           lane * 16);
        }
    }
    if (lane < SC::kScaleU32) {
        pipe_copy<4>(&s_sc[lane],
                     scale_row + static_cast<std::int64_t>(slab) * (SC::kScaleU32 * 4) + lane * 4);
    }
    pipe_commit();
}

// x0 points at the column tile base (x + col0*k); xslab is the slab's first
// K-value. Requires k % 8 == 0 and 16-byte aligned x.
template <class SC, int kTt>
__device__ __forceinline__ void
q5_simt_consume_slab(const __nv_bfloat16* __restrict__ x0, std::int64_t xslab, std::int32_t k,
                     int ncols, const uint4* __restrict__ s_nib, const uint4* __restrict__ s_hi,
                     const std::uint32_t* __restrict__ s_sc, int lane, float (&acc)[kTt]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float w[8];
        SC::dequant_chunk(s_nib, s_hi, s_sc, c, lane, w);
        const std::int64_t xoff = xslab + c * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            if (tt < ncols) {
                const uint4 xv  = load_vec<uint4>(x0 + static_cast<std::int64_t>(tt) * k + xoff);
                const float2 f0 = bf16x2_bits_to_float2(xv.x);
                const float2 f1 = bf16x2_bits_to_float2(xv.y);
                const float2 f2 = bf16x2_bits_to_float2(xv.z);
                const float2 f3 = bf16x2_bits_to_float2(xv.w);
                acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
                acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
                acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
                acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
                acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
                acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
                acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
                acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
            }
        }
    }
}

// One dequant+accumulate step of the direct split2 schedule: 8 K-values of one
// row (the lane's chunk phase) against kTt activation columns.
//   word       : the lane's 4 packed nibble bytes
//   high_bits  : the lane's high-plane byte (raw, pre-inversion)
//   scale_bits : the group's fp16 scale, already broadcast across the group
template <int kTt, int kStride>
__device__ __forceinline__ void
q5_split2_accumulate_chunk(std::uint32_t word, std::uint32_t high_bits, std::uint32_t scale_bits,
                           const __nv_bfloat16* __restrict__ x, std::int64_t xoff,
                           float (&acc)[kTt]) {
    const std::uint32_t hc = high_bits ^ 0xffu;
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    const __half2 bias = __half2half2(__ushort_as_half(0x6410)); // 1040.0
    float w[8];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
        bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
        const __half2 h = __hsub2(half2_from_bits(bits), bias);
        const float2 f  = __half22float2(h);
        w[p]            = f.x * scale;
        w[p + 4]        = f.y * scale;
    }
#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        const uint4 xv  = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kStride + xoff);
        const float2 f0 = bf16x2_bits_to_float2(xv.x);
        const float2 f1 = bf16x2_bits_to_float2(xv.y);
        const float2 f2 = bf16x2_bits_to_float2(xv.z);
        const float2 f3 = bf16x2_bits_to_float2(xv.w);
        acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
        acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
        acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
        acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
        acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
        acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
        acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
        acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
    }
}

template <class SC, int kTt, int kFullSlabs, int kStride, bool SplitOutput = false,
          int SplitRow = 0, bool AddResidual = false>
// Accumulator count scales with the column tile, so the occupancy target has to as well. At 16
// blocks ptxas is capped at 64 registers, which the wide tiles cannot hold: they spill, and the
// spill costs more than the extra warps buy on a kernel this bandwidth-bound. Narrow tiles fit
// inside 64 and prefer the warps.
__launch_bounds__(64, (kTt <= 3 || (kStride <= 6144 && kTt <= 6)) ? 16 : 8) __global__
    void q5_rowsplit_gemm_simt_split2_kernel(const __nv_bfloat16* __restrict__ x,
                                             const std::uint8_t* __restrict__ codes,
                                             const std::uint8_t* __restrict__ high,
                                             const std::uint8_t* __restrict__ scales,
                                             __nv_bfloat16* __restrict__ out, std::int32_t n,
                                             std::int32_t k, std::int32_t t, std::int32_t padded_k,
                                             std::int32_t full_slabs) {
    static_assert(std::is_same_v<SC, Q5RowSplitSimtSchedule>,
                  "direct split2 small-T kernel is Q5-only");
    static_assert(kTt > 0, "direct split2 requires a positive column tile");
    static_assert(kFullSlabs > 0 && kStride > 0, "direct split2 requires exact positive shape");
    (void)full_slabs;
    (void)k;
    (void)t;

    __shared__ float s_part[2][kTt];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int part = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int kg_padded          = padded_k / Q5RowSplitStorage::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * kg_padded *
                                              Q5RowSplitSimtSchedule::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

    // Slab schedule.
    //
    // kFullSlabs is a compile-time constant, so nvcc fully unrolls the K sweep
    // by default. At K=17408 (kFullSlabs=17, kTt=5) that is ~4100 SASS
    // instructions = 64 KB of code executed exactly once per block, which
    // overruns the SM instruction cache and costs real bandwidth. Rolling the
    // loop back up shrinks the body to ~10 KB, but ptxas does not
    // software-pipeline the global loads across the back edge under -rdc=true
    // (which this project builds with), so a naive rolled loop stalls on DRAM
    // latency and is *slower* than the unrolled form. Issuing slab s+1's three
    // weight-plane loads by hand, into registers, before consuming slab s
    // restores the memory-level parallelism explicitly, so both properties hold
    // at once.
    //
    // This only pays when the K sweep is long enough to overrun the instruction
    // cache and the column tile is wide enough that the rolled schedule still
    // saturates DRAM; on the short K=6144 sweep the extra live registers spill
    // instead. Measured on RTX 4090 / sm_89, 5120xK Q5 linear_add, cold L2,
    // repo build flags (-rdc=true, -lineinfo), median of 25-60 launches, us:
    //
    //   K=17408   kTt=2  kTt=3  kTt=4  kTt=5  kTt=8  kTt=12  kTt=14  kTt=16
    //   unrolled   91.0   92.7  100.2  107.5  128.1   192.9   219.1   234.5
    //   rolled     90.8   93.2   98.3  100.5  132.8   167.9   183.3   207.9
    //
    // so the rolled schedule is gated to kFullSlabs > 8 && kTt >= 4; every other
    // instantiation keeps the fully unrolled body unchanged. Deeper prefetch
    // (2-3 slabs) and unroll factors 1/3/4/5 all measured slower. Results are
    // bit-identical either way: the accumulation order does not change.
    //
    // The table above was taken under the previous fixed occupancy target, which
    // capped this kernel at 64 registers. The launch bound is now a function of
    // the column tile, so the wide tiles get 128, and the rolled body's manual
    // prefetch no longer spills. The gate was re-verified against the new bound
    // and still holds, but two rows moved: kTt=8 now favours rolled (116.9 vs
    // 125.9 unrolled) where the table has it losing, and kTt=12/16 are close to
    // a tie rather than a clear rolled win. kTt=3 still prefers unrolled
    // (94.3 vs 101.8), which is what keeps the threshold at 4.
    constexpr bool kRollSlabs = kFullSlabs > 8 && kTt >= 4;

    if constexpr (kRollSlabs) {
        // Lane-invariant slice bases; a slab is a fixed stride off each.
        const std::uint8_t* code_base  = code_row + part * 256 + lane * 4;
        const std::uint8_t* high_base  = high_row + part * 64 + lane;
        const std::uint8_t* scale_base = scale_row + part * 16 + (lane >> 3) * 2;

        std::uint32_t next_word[2];
        std::uint32_t next_high[2];
        std::uint32_t next_scale[2];
        const auto issue_slab = [&](int s) {
#pragma unroll
            for (int local = 0; local < 2; ++local) {
                next_word[local] = *reinterpret_cast<const std::uint32_t*>(
                    code_base + static_cast<std::int64_t>(s) * 512 + local * 128);
                next_high[local] = *(high_base + static_cast<std::int64_t>(s) * 128 + local * 32);
                next_scale[local] = *reinterpret_cast<const std::uint16_t*>(
                    scale_base + static_cast<std::int64_t>(s) * 32 + local * 8);
            }
        };
        issue_slab(0);

#pragma unroll 2
        for (int s = 0; s < kFullSlabs; ++s) {
            std::uint32_t word[2];
            std::uint32_t high_bits[2];
            std::uint32_t scale_bits[2];
#pragma unroll
            for (int local = 0; local < 2; ++local) {
                word[local]       = next_word[local];
                high_bits[local]  = next_high[local];
                scale_bits[local] = next_scale[local];
            }
            if (s + 1 < kFullSlabs) { issue_slab(s + 1); }
#pragma unroll
            for (int local = 0; local < 2; ++local) {
                const int chunk = part * 2 + local;
                q5_split2_accumulate_chunk<kTt, kStride>(
                    word[local], high_bits[local], scale_bits[local], x,
                    static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8, acc);
            }
        }
    } else {
#pragma unroll
        for (int s = 0; s < kFullSlabs; ++s) {
#pragma unroll
            for (int local = 0; local < 2; ++local) {
                const int chunk = part * 2 + local;
                const std::uint8_t* code_phase =
                    code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4;
                const std::uint8_t* high_phase =
                    high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane;
                const int group_in_slab  = chunk * 4 + (lane >> 3);
                std::uint32_t scale_bits = 0;
                if ((lane & 7) == 0) {
                    scale_bits = *reinterpret_cast<const std::uint16_t*>(
                        scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
                }
                scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);
                q5_split2_accumulate_chunk<kTt, kStride>(
                    *reinterpret_cast<const std::uint32_t*>(code_phase),
                    static_cast<std::uint32_t>(*high_phase), scale_bits, x,
                    static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8, acc);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float a = acc[tt];
        a       = warp_reduce_sum(a);
        if (lane == 0) { s_part[part][tt] = a; }
    }

    __syncthreads();

    if (part == 0 && lane < kTt) {
        const std::int64_t index = static_cast<std::int64_t>(lane) * n + row;
        const float sum              = s_part[0][lane] + s_part[1][lane];
        const __nv_bfloat16 sum_bf16 = __float2bfloat16_rn(sum);
        if constexpr (AddResidual) {
            out[index] = __hadd(sum_bf16, out[index]);
        } else {
            out[index] = sum_bf16;
        }
    }
}

// Multi-row split2. The single-row kernel widens every bf16 activation to fp32 immediately before
// its FFMA, because each activation value feeds exactly one weight: SASS shows one PRMT per FFMA,
// 26% of the kernel's instructions. Giving a CTA kRows output rows lets one widened activation feed
// kRows FFMAs, so the conversion and the activation load are both amortised kRows-fold. Everything
// else - the dequant, the accumulate order, the reduction - is unchanged, so results stay
// bit-identical to the single-row kernel row for row.
template <int kRows, int kTt, int kStride>
__device__ __forceinline__ void
q5_split2_accumulate_chunk_rows(const std::uint32_t (&word)[kRows],
                                const std::uint32_t (&high_bits)[kRows],
                                const std::uint32_t (&scale_bits)[kRows],
                                const __nv_bfloat16* __restrict__ x, std::int64_t xoff,
                                float (&acc)[kRows][kTt]) {
    const __half2 bias = __half2half2(__ushort_as_half(0x6410)); // 1040.0
    float w[kRows][8];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const std::uint32_t hc = high_bits[r] ^ 0xffu;
        const float scale =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits[r])));
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word[r] >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(half2_from_bits(bits), bias);
            const float2 f  = __half22float2(h);
            w[r][p]         = f.x * scale;
            w[r][p + 4]     = f.y * scale;
        }
    }
#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        const uint4 xv  = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kStride + xoff);
        const float2 f0 = bf16x2_bits_to_float2(xv.x);
        const float2 f1 = bf16x2_bits_to_float2(xv.y);
        const float2 f2 = bf16x2_bits_to_float2(xv.z);
        const float2 f3 = bf16x2_bits_to_float2(xv.w);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            acc[r][tt] = fmaf(w[r][0], f0.x, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][1], f0.y, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][2], f1.x, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][3], f1.y, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][4], f2.x, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][5], f2.y, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][6], f3.x, acc[r][tt]);
            acc[r][tt] = fmaf(w[r][7], f3.y, acc[r][tt]);
        }
    }
}

template <class SC, int kRows, int kTt, int kFullSlabs, int kStride, bool AddResidual = false>
__launch_bounds__(64, 8) __global__
    void q5_rowsplit_gemm_simt_split2_rows_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ high,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out, std::int32_t n,
                                                  std::int32_t k, std::int32_t t,
                                                  std::int32_t padded_k, std::int32_t full_slabs) {
    static_assert(std::is_same_v<SC, Q5RowSplitSimtSchedule>, "multi-row split2 is Q5-only");
    static_assert(kRows >= 2, "multi-row split2 needs at least two rows");
    (void)full_slabs;
    (void)k;
    (void)t;

    __shared__ float s_part[2][kRows][kTt];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int part = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = static_cast<int>(blockIdx.x) * kRows;
    if (row0 >= n) { return; }
    const int rows_here = min(kRows, n - row0);

    const int kg_padded = padded_k / Q5RowSplitStorage::kGroupK;

    float acc[kRows][kTt];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int i = 0; i < kTt; ++i) { acc[r][i] = 0.0f; }
    }

    const std::uint8_t* code_base[kRows];
    const std::uint8_t* high_base[kRows];
    const std::uint8_t* scale_base[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const std::int64_t rr = row0 + (r < rows_here ? r : 0);
        code_base[r]  = codes + rr * kg_padded * 32 + part * 256 + lane * 4;
        high_base[r]  = high + rr * kg_padded * SC::kHighBytesPerGroup + part * 64 + lane;
        scale_base[r] = scales + rr * kg_padded * 2 + part * 16 + (lane >> 3) * 2;
    }

    std::uint32_t next_word[kRows][2];
    std::uint32_t next_high[kRows][2];
    std::uint32_t next_scale[kRows][2];
    const auto issue_slab = [&](int sl) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int local = 0; local < 2; ++local) {
                next_word[r][local] = *reinterpret_cast<const std::uint32_t*>(
                    code_base[r] + static_cast<std::int64_t>(sl) * 512 + local * 128);
                next_high[r][local] =
                    *(high_base[r] + static_cast<std::int64_t>(sl) * 128 + local * 32);
                next_scale[r][local] = *reinterpret_cast<const std::uint16_t*>(
                    scale_base[r] + static_cast<std::int64_t>(sl) * 32 + local * 8);
            }
        }
    };
    issue_slab(0);

#pragma unroll 1
    for (int sl = 0; sl < kFullSlabs; ++sl) {
        std::uint32_t word[kRows][2];
        std::uint32_t high_bits[kRows][2];
        std::uint32_t scale_bits[kRows][2];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int local = 0; local < 2; ++local) {
                word[r][local]       = next_word[r][local];
                high_bits[r][local]  = next_high[r][local];
                scale_bits[r][local] = next_scale[r][local];
            }
        }
        if (sl + 1 < kFullSlabs) { issue_slab(sl + 1); }
#pragma unroll
        for (int local = 0; local < 2; ++local) {
            const int chunk = part * 2 + local;
            std::uint32_t cw[kRows];
            std::uint32_t ch[kRows];
            std::uint32_t cs[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                cw[r] = word[r][local];
                ch[r] = high_bits[r][local];
                cs[r] = scale_bits[r][local];
            }
            q5_split2_accumulate_chunk_rows<kRows, kTt, kStride>(
                cw, ch, cs, x, static_cast<std::int64_t>(sl) * 1024 + chunk * 256 + lane * 8, acc);
        }
    }

#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            float a = warp_reduce_sum(acc[r][tt]);
            if (lane == 0) { s_part[part][r][tt] = a; }
        }
    }

    __syncthreads();

    if (part == 0 && lane < kTt * kRows) {
        const int r  = lane / kTt;
        const int tt = lane - r * kTt;
        if (r < rows_here) {
            const std::int64_t index     = static_cast<std::int64_t>(tt) * n + row0 + r;
            const float sum              = s_part[0][r][tt] + s_part[1][r][tt];
            const __nv_bfloat16 sum_bf16 = __float2bfloat16_rn(sum);
            if constexpr (AddResidual) {
                out[index] = __hadd(sum_bf16, out[index]);
            } else {
                out[index] = sum_bf16;
            }
        }
    }
}

struct Q5Split4StoreEpilogue {
    template <bool SplitOutput, int SplitRow, int Tokens>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t n, std::int32_t out_ld,
               std::int32_t row, const float (&values)[Tokens]) const {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(token) * out_ld + row] =
                        __float2bfloat16(values[token]);
                } else {
                    out_tail[static_cast<std::int64_t>(token) * (n - SplitRow) + row - SplitRow] =
                        __float2bfloat16(values[token]);
                }
            } else {
                out[static_cast<std::int64_t>(token) * out_ld + row] =
                    __float2bfloat16(values[token]);
            }
        }
    }
};

template <class SC, int kTt, int kFullSlabs, int kStride, bool SplitOutput = false,
          int SplitRow = 0, class Epilogue = Q5Split4StoreEpilogue, bool TriggerPdl = false,
          bool JoinPdl = false>
// Same trade as the split2 bound above: the wide column tiles need more registers than a 10-block
// target leaves them, and the spill costs more than the extra warps return.
__launch_bounds__(128, kTt <= 5 ? 10 : 8) __global__ void q5_rowsplit_gemm_simt_split4_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ out_tail, std::int32_t n,
    std::int32_t out_ld, std::int32_t k, std::int32_t t, std::int32_t padded_k,
    std::int32_t full_slabs, Epilogue epilogue = {}) {
    static_assert(std::is_same_v<SC, Q5RowSplitSimtSchedule>,
                  "direct split4 small-T kernel is Q5-only");
    static_assert(kTt > 0, "direct split4 requires a positive column tile");
    static_assert(kFullSlabs > 0 && kStride > 0, "direct split4 requires exact positive shape");
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q5 split4 requires a positive compile-time seam");
    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }
    (void)full_slabs;
    (void)k;
    (void)t;

    __shared__ float s_part[4][kTt];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(threadIdx.x) >> 5;
    const int row   = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int kg_padded          = padded_k / Q5RowSplitStorage::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * kg_padded *
                                              Q5RowSplitSimtSchedule::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int s = 0; s < kFullSlabs; ++s) {
        const std::uint8_t* code_phase =
            code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4;
        const std::uint8_t* high_phase =
            high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane;
        const int group_in_slab  = chunk * 4 + (lane >> 3);
        std::uint32_t scale_bits = 0;
        if ((lane & 7) == 0) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);

        const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(code_phase);
        const std::uint32_t hc   = static_cast<std::uint32_t>(*high_phase) ^ 0xffu;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
        float w[8];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(half2_from_bits(bits), bias);
            const float2 f  = __half22float2(h);
            w[p]            = f.x * scale;
            w[p + 4]        = f.y * scale;
        }

        const std::int64_t xoff = static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const uint4 xv  = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kStride + xoff);
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
            acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
            acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
            acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
            acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
            acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
            acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
            acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float a = acc[tt];
        a       = warp_reduce_sum(a);
        if (lane == 0) { s_part[chunk][tt] = a; }
    }

    __syncthreads();

    if constexpr (std::is_same_v<Epilogue, Q5Split4StoreEpilogue>) {
        if (chunk == 0 && lane < kTt) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) { sum += s_part[p][lane]; }
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(lane) * out_ld + row] = __float2bfloat16(sum);
                } else {
                    out_tail[static_cast<std::int64_t>(lane) * (n - SplitRow) + row - SplitRow] =
                        __float2bfloat16(sum);
                }
            } else {
                out[static_cast<std::int64_t>(lane) * out_ld + row] = __float2bfloat16(sum);
            }
        }
    } else {
        if (chunk == 0 && lane < kTt) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) { sum += s_part[p][lane]; }
            s_part[0][lane] = sum;
        }
        __syncthreads();
        if (chunk == 0 && lane == 0) {
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, n, out_ld, row,
                                                                s_part[0]);
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

// Multi-row split4. Same trade as the split2 variant above: one widened activation feeds kRows
// FFMAs instead of one, so the bf16->fp32 PRMT that SASS shows paired 1:1 with every FFMA amortises
// kRows-fold. The four-way K split, the dequant and the accumulate order are unchanged, so each row
// produces bit-identical results to the single-row kernel. The split-output seam is evaluated per
// row, so a CTA whose rows straddle it still writes each row to the correct destination.
template <class SC, int kRows, int kTt, int kFullSlabs, int kStride, bool SplitOutput = false,
          int SplitRow = 0, class Epilogue = Q5Split4StoreEpilogue>
__launch_bounds__(128, kTt >= 8 ? 6 : 8) __global__ void q5_rowsplit_gemm_simt_split4_rows_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ out_tail, std::int32_t n,
    std::int32_t out_ld, std::int32_t k, std::int32_t t, std::int32_t padded_k,
    std::int32_t full_slabs, Epilogue epilogue = {}) {
    static_assert(std::is_same_v<SC, Q5RowSplitSimtSchedule>, "multi-row split4 is Q5-only");
    static_assert(kRows >= 2, "multi-row split4 needs at least two rows");
    (void)full_slabs;
    (void)k;
    (void)t;

    __shared__ float s_part[4][kRows][kTt];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(threadIdx.x) >> 5;
    const int row0  = static_cast<int>(blockIdx.x) * kRows;
    if (row0 >= n) { return; }
    const int rows_here = min(kRows, n - row0);

    const int kg_padded = padded_k / Q5RowSplitStorage::kGroupK;

    float acc[kRows][kTt];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int i = 0; i < kTt; ++i) { acc[r][i] = 0.0f; }
    }

#pragma unroll
    for (int s = 0; s < kFullSlabs; ++s) {
        const int group_in_slab = chunk * 4 + (lane >> 3);
        std::uint32_t word[kRows];
        std::uint32_t high_bits[kRows];
        std::uint32_t scale_bits[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const std::int64_t rr = row0 + (r < rows_here ? r : 0);
            const std::uint8_t* code_row = codes + rr * kg_padded * 32;
            const std::uint8_t* high_row =
                high + rr * kg_padded * Q5RowSplitSimtSchedule::kHighBytesPerGroup;
            const std::uint8_t* scale_row = scales + rr * kg_padded * 2;
            word[r] = *reinterpret_cast<const std::uint32_t*>(
                code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4);
            high_bits[r] = static_cast<std::uint32_t>(
                *(high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane));
            std::uint32_t sb = 0;
            if ((lane & 7) == 0) {
                sb = *reinterpret_cast<const std::uint16_t*>(
                    scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
            }
            scale_bits[r] = __shfl_sync(0xffffffffu, sb, lane & ~7);
        }
        q5_split2_accumulate_chunk_rows<kRows, kTt, kStride>(
            word, high_bits, scale_bits, x,
            static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8, acc);
    }

#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            float a = warp_reduce_sum(acc[r][tt]);
            if (lane == 0) { s_part[chunk][r][tt] = a; }
        }
    }

    __syncthreads();

    if constexpr (std::is_same_v<Epilogue, Q5Split4StoreEpilogue>) {
        if (chunk == 0 && lane < kTt * kRows) {
            const int r  = lane / kTt;
            const int tt = lane - r * kTt;
            if (r < rows_here) {
                float sum = 0.0f;
#pragma unroll
                for (int p = 0; p < 4; ++p) { sum += s_part[p][r][tt]; }
                const int row = row0 + r;
                if constexpr (SplitOutput) {
                    if (row < SplitRow) {
                        out[static_cast<std::int64_t>(tt) * out_ld + row] = __float2bfloat16(sum);
                    } else {
                        out_tail[static_cast<std::int64_t>(tt) * (n - SplitRow) + row - SplitRow] =
                            __float2bfloat16(sum);
                    }
                } else {
                    out[static_cast<std::int64_t>(tt) * out_ld + row] = __float2bfloat16(sum);
                }
            }
        }
    } else {
        if (chunk == 0 && lane < kTt * kRows) {
            const int r  = lane / kTt;
            const int tt = lane - r * kTt;
            float sum    = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) { sum += s_part[p][r][tt]; }
            s_part[0][r][tt] = sum;
        }
        __syncthreads();
        if (chunk == 0 && lane < kRows) {
            if (lane < rows_here) {
                epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, n, out_ld,
                                                                    row0 + lane, s_part[0][lane]);
            }
        }
    }
}

// full_slabs is computed on the host: k/1024 when k % 8 == 0 and x is 16-byte
// aligned, else 0 (everything runs through the scalar tail).
template <class SC, int kTt, int kRowsPerBlock, int kStages, bool SplitOutput = false,
          int SplitRow = 0>
__global__ void q5_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                             const std::uint8_t* __restrict__ codes,
                                             const std::uint8_t* __restrict__ high,
                                             const std::uint8_t* __restrict__ scales,
                                             __nv_bfloat16* __restrict__ out,
                                             __nv_bfloat16* __restrict__ out_tail, std::int32_t n,
                                             std::int32_t out_ld, std::int32_t k, std::int32_t t,
                                             std::int32_t padded_k, std::int32_t full_slabs) {
    using Codec                = typename SC::Codec;
    constexpr int kPrefetch    = kStages - 1;
    constexpr int kHighU4Alloc = SC::kHighU4 > 0 ? SC::kHighU4 : 1;
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q5 SIMT requires a positive compile-time seam");

    __shared__ __align__(16) uint4 s_nib[kRowsPerBlock][kStages][SC::kNibU4];
    __shared__ __align__(16) uint4 s_hi[kRowsPerBlock][kStages][kHighU4Alloc];
    __shared__ __align__(16) std::uint32_t s_sc[kRowsPerBlock][kStages][SC::kScaleU32];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    if (row >= n) { return; }
    const int col0  = static_cast<int>(blockIdx.y) * kTt;
    const int ncols = min(kTt, t - col0);

    const int kg_padded          = padded_k / Codec::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row =
        SC::kHighBytesPerGroup > 0
            ? high + static_cast<std::int64_t>(row) * kg_padded * SC::kHighBytesPerGroup
            : nullptr;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;
    const __nv_bfloat16* x0       = x + static_cast<std::int64_t>(col0) * k;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < full_slabs) {
            q5_simt_issue_slab<SC>(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row, high_row,
                                   scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int s = 0; s < full_slabs; ++s) {
        const int fetch = s + kPrefetch;
        if (fetch < full_slabs) {
            const int buf = fetch % kStages;
            q5_simt_issue_slab<SC>(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row,
                                   high_row, scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = s % kStages;
        q5_simt_consume_slab<SC, kTt>(x0, static_cast<std::int64_t>(s) * 1024, k, ncols,
                                      s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], lane,
                                      acc);
        __syncwarp();
    }

    // Scalar tail: remaining groups read global memory directly, masked at k.
    const int g0      = (full_slabs * 1024) / Codec::kGroupK;
    const int kg_used = div_up(k, Codec::kGroupK);
    for (int g = g0; g < kg_used; ++g) {
        const int kk = g * Codec::kGroupK + lane * 2;
        if (kk >= k) { continue; }
        float w0 = 0.0f;
        float w1 = 0.0f;
        Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row) * kg_padded + g, lane,
                         w0, w1);
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            if (tt < ncols) {
                const std::int64_t xb = static_cast<std::int64_t>(tt) * k + kk;
                acc[tt]               = fmaf(w0, __bfloat162float(x0[xb]), acc[tt]);
                if (kk + 1 < k) { acc[tt] = fmaf(w1, __bfloat162float(x0[xb + 1]), acc[tt]); }
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        if (tt >= ncols) { continue; }
        float a = acc[tt];
        a       = warp_reduce_sum(a);
        if (lane == 0) {
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(col0 + tt) * out_ld + row] = __float2bfloat16(a);
                } else {
                    out_tail[static_cast<std::int64_t>(col0 + tt) * (n - SplitRow) + row -
                             SplitRow] = __float2bfloat16(a);
                }
            } else {
                out[static_cast<std::int64_t>(col0 + tt) * out_ld + row] = __float2bfloat16(a);
            }
        }
    }
}

} // namespace ninfer::ops::detail

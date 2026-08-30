#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 8;
    static constexpr int kScaleBytesPerGroup = 2;

    // A chunk is the eight codes one thread decodes, so it spans four packed bytes.
    static constexpr int kCodeBytesPerChunk = 4;
    static constexpr int kChunksPerGroup    = kCodeBytesPerGroup / kCodeBytesPerChunk;
    static constexpr int kHighBytesPerChunk = kHighBytesPerGroup / kChunksPerGroup;
    static_assert(kChunksPerGroup * kCodeBytesPerChunk == kCodeBytesPerGroup &&
                      kHighBytesPerChunk * kChunksPerGroup == kHighBytesPerGroup,
                  "a group's codes and high-bit plane must divide evenly across its chunks");
};

struct Q5ScalarDecodeAtom {
    static constexpr int kGroupK = Q5RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        decode_pair(codes, high, scales, group_index, lane, w0, w1);
    }

    __device__ static __forceinline__ void
    decode_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
                std::int64_t group_index, int lane, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup)));
        const std::uint8_t packed =
            codes[group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        w0 = static_cast<float>(q0) * scale;
        w1 = static_cast<float>(q1) * scale;
    }
};

struct Q5SimtDecodeAtom {
    __device__ static __forceinline__ void decode_eight(std::uint32_t packed,
                                                        std::uint8_t high_bits,
                                                        std::uint16_t scale_bits,
                                                        float (&weights)[8]) {
        const std::uint32_t high = static_cast<std::uint32_t>(high_bits) ^ 0xffu;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((high >> pair) & 1u) << 4) | (((high >> (pair + 4)) & 1u) << 20);
            const __half2 decoded = __hsub2(half2_from_bits(bits), bias);
            const float2 values   = __half22float2(decoded);
            weights[pair]         = values.x * scale;
            weights[pair + 4]     = values.y * scale;
        }
    }
};

struct Q5MmaDecodeAtom {
    // Four packed bytes plus their high-bit byte -> four bf16 pairs, in weight order; out[i] holds
    // the pair decode_pair_with_scale produces at lane = 4 * chunk + i. The five-bit field is sign
    // extended by (v ^ 0x10) - 0x10, which is what bfe.s32 yields there, so the arithmetic and the
    // __floats2bfloat162_rn rounding are the same.
    static __device__ __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        const unsigned high = high_chunk[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned byte        = (word >> (8 * i)) & 0xffu;
            const int q0               = ((static_cast<int>(byte & 0x0fu) |
                                           static_cast<int>(((high >> (2 * i)) & 1u) << 4)) ^
                                          0x10) -
                             0x10;
            const int q1               = ((static_cast<int>(byte >> 4) |
                                           static_cast<int>(((high >> (2 * i + 1)) & 1u) << 4)) ^
                                          0x10) -
                             0x10;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                               static_cast<float>(q1) * scale);
            out[i]                     = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* staged_codes,
                                                                 const std::uint8_t* staged_high,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t staged_group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        return decode_pair_with_scale(staged_codes, staged_high, scale, staged_group_index, lane);
    }

    static __device__ __forceinline__ __nv_bfloat162
    decode_pair_with_scale(const std::uint8_t* staged_codes, const std::uint8_t* staged_high,
                           float scale, std::int64_t staged_group_index, int lane) {
        const std::uint32_t packed = static_cast<std::uint32_t>(
            staged_codes[staged_group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane]);
        const std::uint32_t high_byte = static_cast<std::uint32_t>(
            staged_high[staged_group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)]);
        const int shift = (lane & 3) * 2;
        const std::uint32_t raw0 = (packed & 0x0fu) | (((high_byte >> shift) & 1u) << 4);
        const std::uint32_t raw1 = (packed >> 4) | (((high_byte >> (shift + 1)) & 1u) << 4);
        int q0, q1;
        asm("bfe.s32 %0, %1, 0, 5;" : "=r"(q0) : "r"(raw0));
        asm("bfe.s32 %0, %1, 0, 5;" : "=r"(q1) : "r"(raw1));
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ninfer::ops::detail

#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q4RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;

    // A chunk is the eight codes one thread decodes, so it spans four packed bytes.
    static constexpr int kCodeBytesPerChunk = 4;
    static constexpr int kChunksPerGroup    = kCodeBytesPerGroup / kCodeBytesPerChunk;
    static_assert(kChunksPerGroup * kCodeBytesPerChunk == kCodeBytesPerGroup,
                  "a group's codes must divide evenly into chunks");
};

struct Q4SimtDecodeAtom {
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[8]) {
        const std::uint32_t word = packed ^ 0x88888888u;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const __half2 decoded    = __hsub2(half2_from_bits(bits), bias);
            const float2 values      = __half22float2(decoded);
            weights[pair]            = values.x * scale;
            weights[pair + 4]        = values.y * scale;
        }
    }

    __device__ static __forceinline__ void
    decode_pair(std::uint8_t packed, std::uint16_t scale_bits, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int q0      = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1      = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        w0                = static_cast<float>(q0) * scale;
        w1                = static_cast<float>(q1) * scale;
    }
};

struct Q4MmaDecodeAtom {
    // Four packed bytes -> four bf16 pairs, in weight order; out[i] holds the pair
    // decode_pair_with_scale produces at lane = 4 * chunk + i. A nibble n decodes to
    // (n ^ 8) - 8, which is what bfe.s32 of a 4-bit field yields, so the sign extension, the
    // float multiply by the group scale and the __floats2bfloat162_rn rounding are the same.
    static __device__ __forceinline__ void decode_eight(unsigned word, float scale,
                                                        unsigned (&out)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned byte        = (word >> (8 * i)) & 0xffu;
            const int q0               = (static_cast<int>(byte & 0x0fu) ^ 0x08) - 0x08;
            const int q1               = (static_cast<int>(byte >> 4) ^ 0x08) - 0x08;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                               static_cast<float>(q1) * scale);
            out[i]                     = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        return decode_pair_with_scale(codes, scale, group_index, lane);
    }

    static __device__ __forceinline__ __nv_bfloat162
    decode_pair_with_scale(const std::uint8_t* codes, float scale, std::int64_t group_index,
                           int lane) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + lane]);
        int q0, q1;
        asm("bfe.s32 %0, %1, 0, 4;" : "=r"(q0) : "r"(packed));
        asm("bfe.s32 %0, %1, 4, 4;" : "=r"(q1) : "r"(packed));
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ninfer::ops::detail

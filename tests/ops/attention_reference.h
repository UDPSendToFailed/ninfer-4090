#pragma once

#include "ops/op_tester.h"

#include <array>
#include <cmath>
#include <numeric>

namespace ninfer::test::attention_reference {
inline constexpr int head_dim = 256, q_heads = 24, kv_heads = 4, group = 64;
enum class Format { Bf16, Int8, Rk4v4E8 };

// Independent normalized Sylvester H64 in double; H64 is its own inverse.
inline void hadamard(double* values) {
    for (int step = 1; step < 64; step *= 2) {
        for (int base = 0; base < 64; base += 2 * step) {
            for (int i = 0; i < step; ++i) {
                const double a = values[base + i], b = values[base + step + i];
                values[base + i] = a + b;
                values[base + step + i] = a - b;
            }
        }
    }
    for (int i = 0; i < 64; ++i) { values[i] *= 0.125; }
}
inline std::size_t page_index(int dim, int page, int head, int offset, int d) {
    return d + static_cast<std::size_t>(dim) * (offset + 64 * (head + kv_heads * page));
}
struct EncodedCache {
    Format format;
    int keys, pages, code_dim;
    std::vector<std::uint8_t> k, v;
    std::vector<std::uint16_t> ks, vs;
    std::vector<double> decoded_k, decoded_v;
    std::vector<std::int32_t> table;

    EncodedCache(Format f, int count) : format(f), keys(count), pages((count + 63) / 64),
        code_dim(f == Format::Rk4v4E8 ? 128 : 256),
        k(static_cast<std::size_t>(code_dim) * 64 * kv_heads * pages * (f == Format::Bf16 ? 2 : 1)),
        v(k.size()), ks(static_cast<std::size_t>(4) * 64 * kv_heads * pages, 0x2400), vs(ks),
        decoded_k(static_cast<std::size_t>(keys) * kv_heads * head_dim), decoded_v(decoded_k.size()),
        table(pages) {
        std::iota(table.begin(), table.end(), 0);
        std::uint32_t random = 173;
        auto next = [&] { random = random * 1664525u + 1013904223u; return random; };
        for (int key = 0; key < keys; ++key) {
            for (int h = 0; h < kv_heads; ++h) {
                for (int which = 0; which < 2; ++which) {
                    auto& codes = which ? v : k;
                    auto& decoded = which ? decoded_v : decoded_k;
                    const auto row = static_cast<std::size_t>(head_dim) * (h + kv_heads * key);
                    std::array<int, head_dim> values{};
                    for (auto& value : values) {
                        value = f == Format::Rk4v4E8 ? static_cast<int>(next() % 15) - 7 :
                                static_cast<int>(next() % 255) - 127;
                    }
                    if (f == Format::Rk4v4E8 && !which) {
                        // Integral D8 coset: each 8-vector has even coordinate sum.
                        // Cache decoding tests observable stored codes, not a GPU encoder.
                        for (int d = 0; d < head_dim; d += 8) {
                            const int sum = std::accumulate(values.begin() + d, values.begin() + d + 8, 0);
                            if (sum % 2) { values[d + 7] += values[d + 7] == 7 ? -1 : 1; }
                        }
                    }
                    for (int d = 0; d < head_dim; ++d) {
                        const double value = values[d] * (f == Format::Rk4v4E8 ? 0.03125 : 0.015625);
                        decoded[row + d] = value;
                        if (f == Format::Bf16) {
                            const auto bits = f32_to_bf16(static_cast<float>(value));
                            const auto i = page_index(256, key / 64, h, key % 64, d) * 2;
                            codes[i] = static_cast<std::uint8_t>(bits);
                            codes[i + 1] = static_cast<std::uint8_t>(bits >> 8);
                        } else if (f == Format::Int8) {
                            codes[page_index(256, key / 64, h, key % 64, d)] =
                                static_cast<std::uint8_t>(values[d]);
                        } else {
                            auto& packed = codes[page_index(128, key / 64, h, key % 64, d / 2)];
                            packed |= static_cast<std::uint8_t>((values[d] & 15) << (4 * (d % 2)));
                        }
                    }
                    if (f == Format::Rk4v4E8) {
                        for (int d = 0; d < head_dim; d += group) { hadamard(decoded.data() + row + d); }
                    }
                }
            }
        }
        // Exact powers of two: FP16 0x2400=1/64 and 0x2800=1/32.
        if (f == Format::Rk4v4E8) { std::fill(ks.begin(), ks.end(), 0x2800); vs = ks; }
    }
};
inline std::vector<std::uint16_t> queries(int tokens) {
    std::vector<float> q(static_cast<std::size_t>(tokens) * head_dim * q_heads);
    fill_uniform(q, 818, -0.25f, 0.25f);
    std::vector<std::uint16_t> result(q.size());
    std::transform(q.begin(), q.end(), result.begin(), f32_to_bf16);
    return result;
}
inline std::vector<double> oracle(const EncodedCache& cache,
                                  const std::vector<std::uint16_t>& q,
                                  const std::vector<std::int32_t>& positions,
                                  const std::vector<int>& accessed_keys = {}) {
    std::vector<double> output(q.size(), 0), scores(cache.keys);
    for (std::size_t token = 0; token < positions.size(); ++token) {
        for (int h = 0; h < q_heads; ++h) {
            const auto qr = head_dim * (h + q_heads * token);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < cache.keys; ++key) {
                if (key > positions[token] || (!accessed_keys.empty() &&
                    !std::binary_search(accessed_keys.begin(), accessed_keys.end(), key))) {
                    scores[key] = -std::numeric_limits<double>::infinity(); continue;
                }
                const auto kr = static_cast<std::size_t>(head_dim) * (h / 6 + kv_heads * key);
                double score = 0;
                for (int d = 0; d < head_dim; ++d) { score += bf16_to_f32(q[qr + d]) * cache.decoded_k[kr + d]; }
                scores[key] = score * 0.0625;
                maximum = std::max(maximum, scores[key]);
            }
            if (!std::isfinite(maximum)) { continue; }
            double denominator = 0;
            for (int key = 0; key < cache.keys; ++key) {
                if (!std::isfinite(scores[key])) { continue; }
                const double weight = std::exp(scores[key] - maximum);
                denominator += weight;
                const auto vr = static_cast<std::size_t>(head_dim) * (h / 6 + kv_heads * key);
                for (int d = 0; d < head_dim; ++d) { output[qr + d] += weight * cache.decoded_v[vr + d]; }
            }
            for (int d = 0; d < head_dim; ++d) { output[qr + d] /= denominator; }
        }
    }
    return output;
}
} // namespace ninfer::test::attention_reference

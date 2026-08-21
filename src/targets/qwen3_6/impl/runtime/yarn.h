#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

struct YarnConfig {
    float scale_factor                = 1.0F;
    float attn_temperature_multiplier = 1.0F;
    std::array<float, 32> inv_freq{};
};

inline YarnConfig compute_yarn_config(std::uint32_t context_capacity) {
    constexpr float kNativeContext = 1048576.0F;
    constexpr float kBaseTheta     = 1.0e7F;
    constexpr float kBeta          = 32.0F;
    constexpr float kAlpha         = 1.0F;
    constexpr float kRlow          = kNativeContext / kBeta;  // 32768.0F
    constexpr float kRhigh         = kNativeContext / kAlpha; // 1048576.0F
    constexpr double kPi           = 3.14159265358979323846;

    YarnConfig cfg;
    if (static_cast<float>(context_capacity) > kNativeContext) {
        cfg.scale_factor = static_cast<float>(context_capacity) / kNativeContext;
        // Per YaRN (Peng et al., Eq. 19), when scaling the final dot-product scalar attn_scale
        // directly (instead of Q and K individually), the dot-product multiplier is m^2:
        // AttnScale' = AttnScale * (0.1 * ln(F) + 1.0)
        cfg.attn_temperature_multiplier = 0.1F * std::log(cfg.scale_factor) + 1.0F;
    } else {
        cfg.scale_factor                = 1.0F;
        cfg.attn_temperature_multiplier = 1.0F;
    }

    for (int i = 0; i < 32; ++i) {
        const float exp       = -2.0F * static_cast<float>(i) / 64.0F;
        const float base_freq = std::pow(kBaseTheta, exp);
        if (cfg.scale_factor <= 1.0F) {
            cfg.inv_freq[i] = base_freq;
        } else {
            const double wavelength = (2.0 * kPi) / static_cast<double>(base_freq);
            float gamma             = 0.0F;
            if (wavelength > static_cast<double>(kRhigh)) {
                gamma = 1.0F;
            } else if (wavelength > static_cast<double>(kRlow)) {
                gamma = static_cast<float>((wavelength - static_cast<double>(kRlow)) /
                                           static_cast<double>(kRhigh - kRlow));
            }
            cfg.inv_freq[i] = base_freq * ((1.0F - gamma) + gamma / cfg.scale_factor);
        }
    }
    return cfg;
}

} // namespace ninfer::targets::qwen3_6::detail


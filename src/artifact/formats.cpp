#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16_CTRL, std::string_view{"bf16"}},
    std::pair{QType::FP32_CTRL, std::string_view{"fp32"}},
    std::pair{QType::I32_CTRL, std::string_view{"int32"}},
    std::pair{QType::Q4G64_F16S, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5G64_F16S, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6G64_F16S, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::W8G32_F16S, std::string_view{"q8_g32_fp16"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninfer::artifact

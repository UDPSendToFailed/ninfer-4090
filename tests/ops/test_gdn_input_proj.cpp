#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 GDN-input-projection Op.
constexpr ReductionCriterion kGdnInputProjA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kGdnInputProjA4Tolerance{0.16, 4.0e-3, 0.16};

int verify_output_range(std::string_view label, const GuardedBf16Tensor& output,
                        std::int32_t full_rows, std::int32_t output_row_offset,
                        std::int32_t output_rows, const quantized_weight::PackedWeight& weight,
                        std::int32_t weight_row_offset, const std::vector<float>& activation,
                        std::int32_t hidden, std::int32_t tokens) {
    const std::vector<double> actual =
        gather_rows(output.values(), full_rows, output_row_offset, output_rows, tokens);
    const std::vector<double> expected =
        projection_oracle(weight, weight_row_offset, output_rows, activation, hidden, tokens);
    return compare(label, actual, expected, kGdnInputProjA16Tolerance);
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkRows      = 4096;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkRows + kValueRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 401U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output   = qkv.tensor();
    Tensor z_output = z.tensor();
    ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q4/Q5 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qk" + suffix, qkv, kRows, 0, kQkRows, query_key.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn value" + suffix, qkv, kRows, kQkRows, kValueRows,
                                    value_z_weight.host, 0, activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, value_z_weight.host,
                                    kValueRows, activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 409U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 419U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 16, 17}) {
        failures += run_q4_q5_case(query_key, value_z_weight, tokens);
    }
    return failures;
}

int run_w8_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQkvRows     = 8192;
    constexpr std::int32_t kZRows       = 4096;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 501U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qkv" + suffix, qkv, kQkvRows, 0, kQkvRows, parent.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 503U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 97}) { failures += run_w8_case(parent, tokens); }
    return failures;
}

int verify_output_range_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                std::int32_t full_rows, std::int32_t output_row_offset,
                                std::int32_t output_rows,
                                const quantized_weight::PackedWeight& weight,
                                std::int32_t weight_row_offset,
                                const std::vector<float>& activation, std::int32_t hidden,
                                std::int32_t tokens, const ReductionCriterion& criterion) {
    const std::vector<double> values     = output.values();
    const std::vector<std::int32_t> rows = sampled_rows(output_rows);
    std::vector<std::int32_t> selected_tokens;
    for (const std::int32_t token :
         {0, 1, tokens / 4, tokens / 2, (3 * tokens) / 4, tokens - 2, tokens - 1}) {
        if (token >= 0 && token < tokens &&
            std::find(selected_tokens.begin(), selected_tokens.end(), token) ==
                selected_tokens.end()) {
            selected_tokens.push_back(token);
        }
    }
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * selected_tokens.size());
    expected.reserve(rows.size() * selected_tokens.size());
    for (const std::int32_t local_row : rows) {
        const std::int32_t output_row = output_row_offset + local_row;
        const std::int32_t weight_row = weight_row_offset + local_row;
        for (const std::int32_t token : selected_tokens) {
            actual.push_back(values[static_cast<std::size_t>(token) * full_rows + output_row]);
            expected.push_back(quantized_weight::dot_fp64(
                weight, weight_row, activation.data() + static_cast<std::size_t>(token) * hidden,
                hidden));
        }
    }
    return compare(label, actual, expected, criterion);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_w8();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj\n";
    return failures == 0 ? 0 : 1;
}

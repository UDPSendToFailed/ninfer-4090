#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "core/weight_view.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::artifact::Reader;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;

Json normative_directory() {
    return {
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 4096}}})},
        {"components", {{"text", {{"config", {{"hidden_size", 5120}}}}}}},
        {"metadata", {{"name", "fixture-model"}}},
        {"objects", Json::array({
                        {{"id", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw_bytes_v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"id", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "bf16"},
                         {"layout", "contiguous_le_v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"id", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "fp32"},
                         {"layout", "contiguous_le_v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"id", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "int32"},
                         {"layout", "contiguous_le_v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"id", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "q4_g64_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"id", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "q5_g64_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"id", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "q6_g64_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"id", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "q8_g32_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                    })},
        {"bindings", {{"bf16_param", {{"object", "bf16"}}}}},
        {"uses", Json::array()},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::weight_geometry;
    constexpr QuantLayout direct = QuantLayout::Contiguous;
    constexpr QuantLayout rows   = QuantLayout::RowSplit;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};

    if (weight_geometry(QType::BF16_CTRL, direct, shape_2x3).bytes != 12 ||
        weight_geometry(QType::FP32_CTRL, direct, {}).bytes != 4 ||
        weight_geometry(QType::I32_CTRL, direct, shape_2).bytes != 8 ||
        weight_geometry(QType::Q4G64_F16S, rows, q4_shape).bytes != 260 ||
        weight_geometry(QType::Q5G64_F16S, rows, q5_shape).bytes != 528 ||
        weight_geometry(QType::Q6G64_F16S, rows, q6_shape).bytes != 516 ||
        weight_geometry(QType::W8G32_F16S, rows, w8_shape).bytes != 264) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.directory().objects.size() != 8) {
        throw std::runtime_error("unexpected object count");
    }
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        reader.validate_object(ninfer::artifact::ObjectHandle{i});
    }
    const auto res_bytes = reader.read_object(reader.find("resource"));
    if (res_bytes.size() != 3 || res_bytes[0] != std::byte{1}) {
        throw std::runtime_error("unexpected resource payload");
    }
}

void test_rejections() {
    auto v2_fixture = write_fixture(normative_directory(), "v2",
                                    ninfer::test::artifact_fixture::kV2Magic);
    expect_artifact_error([&] { (void)Reader(v2_fixture.path); }, "v2 magic");

    auto missing_text = normative_directory();
    missing_text["components"].erase("text");
    auto missing_text_file = write_fixture(missing_text, "missing_text");
    expect_artifact_error([&] { (void)Reader(missing_text_file.path); }, "missing text component");

    auto bad_size = normative_directory();
    bad_size["objects"][1]["bytes"] = 999;
    auto bad_size_file = write_fixture(bad_size, "bad_size");
    expect_artifact_error([&] {
        Reader r(bad_size_file.path);
        r.validate_object(r.find("bf16"));
    }, "mismatched tensor size");
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_rejections();
        std::cout << "test_ninfer_artifact_reader: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

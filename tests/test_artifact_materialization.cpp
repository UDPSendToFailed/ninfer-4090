#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 12288}}})},
            {"components",
             {{"text",
               {{"config", {{"hidden_size", 4}}},
                {"resources", {{"test.json", "frontend/test.json"}}}}}}},
            {"objects", Json::array({
                            {{"id", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw_bytes_v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"id", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "bf16"},
                             {"layout", "contiguous_le_v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"id", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "bf16"},
                             {"layout", "contiguous_le_v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                        })},
            {"bindings",
             {{"param/test", {{"object", "weights/test"}}},
              {"param/second", {{"object", "weights/second"}}}}},
            {"uses", Json::array()},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);

        ninfer::artifact::Binder validation_binder(reader);
        (void)validation_binder.resource("text", "test.json");
        (void)validation_binder.parameter("param/second", {4}, ninfer::artifact::Residency::Device);
        const auto validation_plan = std::move(validation_binder).finish();
        require(validation_plan.object_count == 3 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "unrequested tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::DeviceContext device(0);
        ninfer::artifact::Binder binder(reader);
        const auto resource_handle = binder.resource("text", "test.json");
        const auto second_ref =
            binder.parameter("param/second", {4}, ninfer::artifact::Residency::Device);
        auto plan         = std::move(binder).finish();
        auto materialized = ninfer::artifact::materialize(reader, std::move(plan), device);

        const auto host_res = materialized.host_bytes(resource_handle);
        require(host_res.size() == kResource.size() &&
                    std::equal(host_res.begin(), host_res.end(), kResource.begin()),
                "materialized host resource bytes mismatch");

        const auto view   = ninfer::artifact::bind_view(second_ref, materialized);
        const auto tensor = ninfer::weight_tensor(view, {4});
        std::array<std::byte, 8> readback{};
        CUDA_CHECK(cudaMemcpy(readback.data(), tensor.data, readback.size(), cudaMemcpyDeviceToHost));
        require(std::equal(readback.begin(), readback.end(), kSecondTensor.begin()),
                "materialized device tensor bytes mismatch");
        require(materialized.stats().h2d_bytes == kSecondTensor.size(),
                "unexpected h2d_bytes count");

        std::cout << "test_artifact_materialization: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

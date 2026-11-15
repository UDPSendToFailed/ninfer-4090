#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    const std::filesystem::path default_path =
        std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
    if (std::filesystem::is_regular_file(default_path)) { return default_path; }
    const std::filesystem::path fallback_llm =
        std::filesystem::path("C:/Users/Henrik/Desktop/LLM") / filename;
    if (std::filesystem::is_regular_file(fallback_llm)) { return fallback_llm; }
    return default_path;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

int verify_v3_bundle(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, profile, all_features());
    if (plan.materialization.device_objects.empty() ||
        plan.materialization.host_objects.empty() ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "v3 materialization plan is incomplete: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_rejection() {
    try {
        (void)Package::resolve_weights({"qwen3.6-27b", "unknown"});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("qwen3.6-27b/unknown") != std::string::npos) { return 0; }
    }
    std::cerr << "unknown weights identity was not rejected with the full identity\n";
    return 1;
}

int verify_profile_mismatch_rejection() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.use_cuda_graph = false;
    auto planner = Package::make_sequence_planner(device, options, WeightsProfile::GroupwiseInt);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    auto sequence             = std::move(planner).finalize(pages);
    RuntimeModelView empty_model;
    try {
        (void)ninfer::targets::qwen3_6::create_program<Variant>(
            empty_model, WeightsProfile::GroupwiseIntW8Endpoints, std::move(sequence), device);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("weights profile") != std::string::npos) { return 0; }
    }
    std::cerr << "mismatched load/sequence weights profiles were not rejected\n";
    return 1;
}

} // namespace

int main() {
    if (const int result = verify_rejection(); result != 0) { return result; }
    if (const int result = verify_profile_mismatch_rejection(); result != 0) { return result; }

    const std::filesystem::path qwen38_latest =
        artifact_path("NINFER_QWEN3_8_27B_LATEST_WEIGHTS", "qwen3_8_27b_latest.ninfer");
    if (std::filesystem::is_regular_file(qwen38_latest)) {
        if (const int result =
                verify_v3_bundle(qwen38_latest, WeightsProfile::GroupwiseIntW8Endpoints);
            result != 0) {
            return result;
        }
        return 0;
    }

    std::cerr << "skip: real v3 27B artifact not found: " << qwen38_latest << '\n';
    return 77;
}

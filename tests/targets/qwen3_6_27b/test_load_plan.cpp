#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    const std::filesystem::path default_path = std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
    if (std::filesystem::is_regular_file(default_path)) {
        return default_path;
    }
    const std::filesystem::path fallback_llm = std::filesystem::path("C:/Users/Henrik/Desktop/LLM") / filename;
    if (std::filesystem::is_regular_file(fallback_llm)) {
        return fallback_llm;
    }
    return default_path;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

std::size_t dflash2_device_objects(const ninfer::artifact::Reader& reader,
                                   const ninfer::artifact::MaterializationPlan& plan) {
    return static_cast<std::size_t>(
        std::ranges::count_if(plan.device_objects, [&](const auto& item) {
            return ninfer::artifact::object_name(reader.objects().at(item.object.index))
                .starts_with("dflash2/");
        }));
}

int verify_legacy_dflash2_compatibility(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, profile, all_features());
    if (dflash2_device_objects(reader, plan.materialization) != 0) {
        std::cerr << "legacy artifact unexpectedly bound DFlash2 on device: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_dflash2_bundle(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, profile, all_features());
    if (dflash2_device_objects(reader, plan.materialization) != 0) {
        std::cerr << "DFlash2 bundle materialized on device instead of ValidateOnly: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_groupwise(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::GroupwiseInt) {
        std::cerr << "groupwise identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::GroupwiseInt, all_features());
    if (plan.materialization.object_count != 1124 ||
        plan.materialization.device_objects.size() != 1118 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "groupwise materialization plan is incomplete\n";
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::Q6G64_F16S ||
        plan.bindings.output_head.format != NumericFormat::Q6G64_F16S) {
        std::cerr << "groupwise vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (layer.is_full_attention) {
            if (!std::holds_alternative<SplitAttentionProjectionPlan>(layer.attention.projection)) {
                std::cerr << "groupwise attention parent boundary changed\n";
                return 1;
            }
        } else if (!std::holds_alternative<SplitGdnInputProjectionPlan>(
                       layer.gdn.input_projection)) {
            std::cerr << "groupwise GDN parent boundary changed\n";
            return 1;
        }
        if (layer.mlp.gate_up.format != NumericFormat::Q4G64_F16S ||
            layer.mlp.down.format != NumericFormat::Q5G64_F16S) {
            std::cerr << "groupwise MLP storage profile changed\n";
            return 1;
        }
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

    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path qwen38_groupwise =
        artifact_path("NINFER_QWEN3_8_27B_WEIGHTS", "qwen3_8_27b.ninfer");
    const std::filesystem::path qwen38_dflash2 =
        artifact_path("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS", "qwen3_8_27b_dflash2.ninfer");

    bool verified_any = false;
    if (std::filesystem::is_regular_file(groupwise)) {
        if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
        if (const int result =
                verify_legacy_dflash2_compatibility(groupwise, WeightsProfile::GroupwiseInt);
            result != 0) {
            return result;
        }
        verified_any = true;
    }

    if (std::filesystem::is_regular_file(qwen38_groupwise)) {
        if (const int result = verify_legacy_dflash2_compatibility(
                qwen38_groupwise, WeightsProfile::GroupwiseIntW8Endpoints);
            result != 0) {
            return result;
        }
        verified_any = true;
    }

    if (std::filesystem::is_regular_file(qwen38_dflash2)) {
        if (const int result = verify_dflash2_bundle(qwen38_dflash2,
                                                     WeightsProfile::GroupwiseIntW8Endpoints);
            result != 0) {
            return result;
        }
        verified_any = true;
    }

    if (!verified_any) {
        std::cerr << "skip: real 27B artifact is required: groupwise=" << groupwise << '\n';
        return 77;
    }
    return 0;
}

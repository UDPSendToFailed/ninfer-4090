#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "artifact/materializer.h"

#include <cstddef>
#include <string>

namespace ninfer::targets::qwen3_6 {
namespace {

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_host_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    FrontendResourcePlan plan{
        .tokenizer_json         = binder.resource("text", "tokenizer.json"),
        .tokenizer_config_json  = binder.resource("text", "tokenizer_config.json"),
        .chat_template_jinja    = binder.resource("text", "chat_template.jinja"),
        .generation_config_json = binder.resource("text", "generation_config.json"),
    };
    if (binder.reader().directory().components.contains("vision")) {
        plan.preprocessor_config_json = binder.resource("vision", "preprocessor_config.json");
        plan.video_preprocessor_config_json =
            binder.resource("vision", "video_preprocessor_config.json");
    }
    return plan;
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    FrontendResources out{
        .tokenizer_json         = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json  = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja    = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json = take_string(materialized, plan.generation_config_json),
    };
    if (plan.preprocessor_config_json) {
        out.preprocessor_config_json = take_string(materialized, *plan.preprocessor_config_json);
    }
    if (plan.video_preprocessor_config_json) {
        out.video_preprocessor_config_json =
            take_string(materialized, *plan.video_preprocessor_config_json);
    }
    return out;
}

} // namespace ninfer::targets::qwen3_6

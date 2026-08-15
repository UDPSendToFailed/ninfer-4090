#include "serve/serve_metrics.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::serve {

namespace {

void append_counter(std::string& out, const char* name, std::uint64_t value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %llu\n", name,
                  static_cast<unsigned long long>(value));
    out += line;
}

void append_counter(std::string& out, const char* name, double value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.6f\n", name, value);
    out += line;
}

} // namespace

void ServeMetrics::record(const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    const std::uint64_t cached = m.prefix_cache_hit_tokens;
    const std::uint64_t prompt = outcome.prompt_tokens > 0
                                     ? static_cast<std::uint64_t>(outcome.prompt_tokens)
                                     : 0;
    const std::uint64_t computed_prefill = prompt > cached ? prompt - cached : 0;

    const std::lock_guard<std::mutex> lock(mutex_);
    requests_total_ += 1;
    prompt_tokens_total_ += computed_prefill;
    prompt_seconds_total_ += std::max(0.0, m.prefill_seconds);
    tokens_predicted_total_ +=
        outcome.completion_tokens > 0 ? static_cast<std::uint64_t>(outcome.completion_tokens) : 0;
    tokens_predicted_seconds_total_ += std::max(0.0, m.decode_seconds);
    prefix_cache_hit_tokens_total_ += cached;
    speculative_draft_tokens_total_ += m.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += m.speculative_accepted_tokens;
}

std::string ServeMetrics::render() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::string out;
    out.reserve(640);
    append_counter(out, "llamacpp:prompt_tokens_total", prompt_tokens_total_);
    append_counter(out, "llamacpp:prompt_seconds_total", prompt_seconds_total_);
    append_counter(out, "llamacpp:tokens_predicted_total", tokens_predicted_total_);
    append_counter(out, "llamacpp:tokens_predicted_seconds_total", tokens_predicted_seconds_total_);
    append_counter(out, "ninfer:requests_total", requests_total_);
    append_counter(out, "ninfer:prefix_cache_hit_tokens_total", prefix_cache_hit_tokens_total_);
    append_counter(out, "ninfer:draft_tokens_total", speculative_draft_tokens_total_);
    append_counter(out, "ninfer:draft_accepted_tokens_total", speculative_accepted_tokens_total_);
    return out;
}

} // namespace ninfer::serve

#pragma once

// Cumulative counters behind GET /metrics, in the flat `name value` subset of
// the Prometheus text format.
//
// The four llamacpp:-prefixed counters reproduce llama.cpp's --metrics
// semantics - computed prefill tokens (prefix-cache hits excluded) billed
// against prefill wall time, committed decode tokens against decode wall
// time - so scrapers that difference llama.cpp counters read this server
// without changes. The ninfer:-prefixed series report what llama.cpp cannot:
// speculative draft/acceptance totals and prefix-cache reuse.

#include "serve/generation_service.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

class ServeMetrics {
public:
    // Accumulates one completed request. Called from the same funnel as the
    // request-done log line, so every protocol and both streaming modes count.
    void record(const GenerationOutcome& outcome);

    // One complete Prometheus text body, without HTTP framing.
    [[nodiscard]] std::string render() const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                    = 0;
    std::uint64_t prompt_tokens_total_               = 0;
    double prompt_seconds_total_                     = 0.0;
    std::uint64_t tokens_predicted_total_            = 0;
    double tokens_predicted_seconds_total_           = 0.0;
    std::uint64_t prefix_cache_hit_tokens_total_     = 0;
    std::uint64_t speculative_draft_tokens_total_    = 0;
    std::uint64_t speculative_accepted_tokens_total_ = 0;
};

} // namespace ninfer::serve

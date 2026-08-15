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
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

class ServeMetrics {
public:
    // In-flight bookkeeping, hooked on the request start/done/error log
    // funnel. Entries are keyed by request id: end is idempotent and a
    // request that errors after starting is removed the same way as one that
    // completes, so no path can leak a permanently busy slot.
    void begin_request(std::uint64_t id, int prompt_tokens);
    void end_request(std::uint64_t id);

    // Oldest-first (id order = FIFO arrival order) prompt sizes of in-flight
    // requests, for /slots. The engine does not expose its own slot table;
    // these are HTTP-layer queue positions, which coincide with engine state
    // for the bounded-FIFO, no-preemption scheduler this server runs.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, int>> active_snapshot() const;

    // Accumulates one completed request. Called from the same funnel as the
    // request-done log line, so every protocol and both streaming modes count.
    void record(const GenerationOutcome& outcome);

    // One complete Prometheus text body, without HTTP framing. In-flight
    // requests are split into processing/deferred against `max_concurrency`,
    // matching the FIFO scheduler's work-conserving behavior.
    [[nodiscard]] std::string render(std::uint32_t max_concurrency) const;

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
    std::map<std::uint64_t, int> active_;
};

} // namespace ninfer::serve

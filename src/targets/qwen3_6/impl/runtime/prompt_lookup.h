#pragma once

#include <ninfer/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct PromptLookupDraft {
    std::uint32_t count = 0;
    std::array<TokenId, 4> tokens{};
};

// Fast context n-gram lookup over the active sequence ledger.
// Checks if the trailing n-gram (e.g. 3 tokens) appears earlier in the sequence context,
// and if so, extracts the subsequent continuation tokens as speculative draft candidates.
inline PromptLookupDraft find_prompt_lookup_draft(std::span<const TokenId> ledger,
                                                 std::uint32_t max_drafts = 3,
                                                 std::uint32_t ngram_size = 3) noexcept {
    PromptLookupDraft result{};
    if (ledger.size() < static_cast<std::size_t>(ngram_size + 1) || max_drafts == 0) {
        return result;
    }

    const std::size_t total = ledger.size();
    const TokenId t0 = ledger[total - 3];
    const TokenId t1 = ledger[total - 2];
    const TokenId t2 = ledger[total - 1];

    // Search backward through sequence ledger for the most recent previous occurrence of (t0, t1, t2).
    // Search up to (total - ngram_size - 1) so that there is at least 1 continuation token.
    const std::size_t search_end = total - ngram_size;
    for (std::size_t i = search_end; i >= ngram_size; --i) {
        const std::size_t pos = i - 1;
        if (ledger[pos - 2] == t0 && ledger[pos - 1] == t1 && ledger[pos] == t2) {
            // Found matching n-gram at pos.
            // Extract following continuation tokens up to max_drafts.
            const std::size_t avail = total - (pos + 1);
            const std::size_t num_to_draft =
                std::min<std::size_t>(static_cast<std::size_t>(max_drafts), avail);
            if (num_to_draft > 0) {
                result.count = static_cast<std::uint32_t>(num_to_draft);
                for (std::size_t d = 0; d < num_to_draft; ++d) {
                    result.tokens[d] = ledger[pos + 1 + d];
                }
                return result;
            }
        }
    }
    return result;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

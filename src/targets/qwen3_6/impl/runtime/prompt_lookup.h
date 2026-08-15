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

// Fast hierarchical context n-gram lookup over the active sequence ledger.
// Tries N=4 match first for maximum continuation confidence,
// then falls back to N=3 and N=2 to capture smaller programming identifiers.
inline PromptLookupDraft find_prompt_lookup_draft(std::span<const TokenId> ledger,
                                                 std::uint32_t max_drafts = 3) noexcept {
    PromptLookupDraft result{};
    const std::size_t total = ledger.size();
    if (total < 3 || max_drafts == 0) {
        return result;
    }

    // 1. Try N=4 match if sufficient history exists
    if (total >= 5) {
        const TokenId t0 = ledger[total - 4];
        const TokenId t1 = ledger[total - 3];
        const TokenId t2 = ledger[total - 2];
        const TokenId t3 = ledger[total - 1];
        const std::size_t search_end = total - 4;
        for (std::size_t i = search_end; i >= 4; --i) {
            const std::size_t pos = i - 1;
            if (ledger[pos - 3] == t0 && ledger[pos - 2] == t1 &&
                ledger[pos - 1] == t2 && ledger[pos] == t3) {
                const std::size_t avail = total - (pos + 1);
                const std::size_t num =
                    std::min<std::size_t>(static_cast<std::size_t>(max_drafts), avail);
                if (num > 0) {
                    result.count = static_cast<std::uint32_t>(num);
                    for (std::size_t d = 0; d < num; ++d) {
                        result.tokens[d] = ledger[pos + 1 + d];
                    }
                    return result;
                }
            }
        }
    }

    // 2. Try N=3 match
    if (total >= 4) {
        const TokenId t0 = ledger[total - 3];
        const TokenId t1 = ledger[total - 2];
        const TokenId t2 = ledger[total - 1];
        const std::size_t search_end = total - 3;
        for (std::size_t i = search_end; i >= 3; --i) {
            const std::size_t pos = i - 1;
            if (ledger[pos - 2] == t0 && ledger[pos - 1] == t1 && ledger[pos] == t2) {
                const std::size_t avail = total - (pos + 1);
                const std::size_t num =
                    std::min<std::size_t>(static_cast<std::size_t>(max_drafts), avail);
                if (num > 0) {
                    result.count = static_cast<std::uint32_t>(num);
                    for (std::size_t d = 0; d < num; ++d) {
                        result.tokens[d] = ledger[pos + 1 + d];
                    }
                    return result;
                }
            }
        }
    }

    // 3. Try N=2 match (fall back for short keywords/punctuation)
    {
        const TokenId t0 = ledger[total - 2];
        const TokenId t1 = ledger[total - 1];
        const std::size_t search_end = total - 2;
        for (std::size_t i = search_end; i >= 2; --i) {
            const std::size_t pos = i - 1;
            if (ledger[pos - 1] == t0 && ledger[pos] == t1) {
                const std::size_t avail = total - (pos + 1);
                const std::size_t num =
                    std::min<std::size_t>(static_cast<std::size_t>(max_drafts), avail);
                if (num > 0) {
                    result.count = static_cast<std::uint32_t>(num);
                    for (std::size_t d = 0; d < num; ++d) {
                        result.tokens[d] = ledger[pos + 1 + d];
                    }
                    return result;
                }
            }
        }
    }

    return result;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

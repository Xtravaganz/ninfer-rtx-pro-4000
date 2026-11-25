#pragma once
#include <cstdint>

// Stage6F is only the host-side execution schedule and batched-visit ABI.
// No model inference, GPU graphs, attention or quantization is implied.
extern "C" {
struct InklingBatchVisit6F {
    std::uint32_t ordinal;
    std::uint32_t physical_layer;
    std::uint32_t cache_slot;
    std::uint32_t core_pass;
    std::uint32_t stage;          // 0 prelude, 1 core, 2 postlude
    std::uint32_t block_boundary;
    std::uint32_t reads_messages;
    std::uint32_t reset_to_prelude;
    std::uint32_t tokens;         // all positions in the requested chunk
};
using InklingVisitCallback6F = int (*)(const InklingBatchVisit6F*, void*);
// Writes a fixed 48-entry schedule; returns count 48 or a negative error.
int inkling6f_plan(InklingBatchVisit6F* visits, std::uint32_t capacity,
                   std::uint32_t tokens) noexcept;
// Calls the callback once per physical visit, carrying the entire token chunk.
// Returns 48, a negative invalid-argument error, or the callback's positive code.
int inkling6f_dispatch(std::uint32_t tokens, InklingVisitCallback6F callback,
                       void* context) noexcept;
}

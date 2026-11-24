#include "models/inkling2/plan.h"

#include <stdexcept>
#include <vector>

namespace ninfer::models::inkling2 {

ExecutionPlan make_execution_plan(const TextConfig& c) {
    if (c.physical_layers != c.sliding.size() || c.core_start != 6 || c.core_end != 18 ||
        c.core_passes != 3 || c.block_layers != 6 || c.varloop) {
        throw std::invalid_argument("Inkling2 Mini: unsupported loop layout");
    }
    const auto core_count = c.core_end - c.core_start;
    const auto total = c.physical_layers + (c.core_passes - 1) * core_count;
    if (total != 48) {
        throw std::invalid_argument("Inkling2 Mini: expected 48 logical cache slots");
    }
    ExecutionPlan out;
    out.cache_slots = total;
    out.steps.reserve(total);
    const auto append = [&](Stage stage, std::uint32_t layer, std::uint32_t pass,
                            std::uint32_t cache_slot, bool reset) {
        out.steps.push_back({stage, layer, pass, cache_slot, c.sliding.at(layer),
                             layer % c.block_layers == 0,
                             stage == Stage::Core && pass > 0, reset});
    };
    // The first 24 HF cache slots belong to the physical layers. Additional
    // passes append their own 12 slots after the physical model's 24 slots.
    for (std::uint32_t layer = 0; layer < c.core_start; ++layer)
        append(Stage::Prelude, layer, 0, layer, false);
    for (std::uint32_t pass = 0; pass < c.core_passes; ++pass) {
        for (std::uint32_t layer = c.core_start; layer < c.core_end; ++layer) {
            const auto slot = pass == 0 ? layer : c.physical_layers + (pass - 1) * core_count +
                                                      (layer - c.core_start);
            append(Stage::Core, layer, pass, slot, pass > 0 && layer == c.core_start);
        }
    }
    for (std::uint32_t layer = c.core_end; layer < c.physical_layers; ++layer)
        append(Stage::Postlude, layer, 0, layer, false);

    if (out.steps.size() != total) throw std::logic_error("Inkling2 schedule length mismatch");
    std::vector<bool> seen(total, false);
    for (const auto& step : out.steps) {
        if (step.cache_slot >= total || seen[step.cache_slot])
            throw std::logic_error("Inkling2 cache slot is duplicated or out of range");
        seen[step.cache_slot] = true;
    }
    return out;
}

} // namespace ninfer::models::inkling2

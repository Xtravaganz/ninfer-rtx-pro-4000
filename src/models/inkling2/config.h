#pragma once

#include "models/inkling2/plan.h"
#include "artifact/schema.h"

namespace ninfer::models::inkling2 {

struct CheckpointContract {
    TextConfig text;
    bool vision_present = false; // recognized, not yet runnable
    bool mtp_present = false;    // recognized, not yet runnable
};

// Parses ORIGINAL Hugging Face config.json (top-level text_config), not an
// NInfer artifact directory. Strictly scoped to Inkling 2 Mini v1.
[[nodiscard]] CheckpointContract parse_hf_config(const artifact::Json& root);

} // namespace ninfer::models::inkling2

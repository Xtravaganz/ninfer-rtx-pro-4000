#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace ninfer::models::inkling2 {

// The 24 checkpoint layers are physical weights. The core is replayed, so each
// execution of a physical core layer needs its own causal/short-conv state.
struct TextConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t physical_layers = 0;
    std::uint32_t vocabulary_rows = 0;
    std::uint32_t public_tokens = 0;
    std::uint32_t eos_token_id = 0;
    std::uint32_t attention_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t experts = 0;
    std::uint32_t top_k = 0;
    std::uint32_t intermediate = 0;
    std::uint32_t sliding_window = 0;
    std::uint32_t convolution_kernel = 0;
    std::uint32_t block_layers = 0;
    std::uint32_t core_start = 0;
    std::uint32_t core_end = 0;
    std::uint32_t core_passes = 0;
    std::uint32_t gate_rank = 0;
    float norm_epsilon = 0;
    float gate_softcap = 0;
    float up_softcap = 0;
    float logit_softcap = 0;
    float route_scale = 0;
    bool gate_bias = false;
    bool varloop = false;
    std::array<bool, 24> sliding{};
};

enum class Stage : std::uint8_t { Prelude, Core, Postlude };

struct ExecutionStep {
    Stage stage;
    std::uint32_t physical_layer;
    std::uint32_t core_pass;       // 0..core_passes-1 in Core; 0 elsewhere
    std::uint32_t cache_slot;      // HF cache layout, NOT execution ordinal
    bool sliding_attention;
    bool block_boundary;
    bool reads_previous_pass_messages;
    bool reset_to_prelude;         // true exactly at start of Core pass 2+
};

struct ExecutionPlan {
    std::vector<ExecutionStep> steps; // execution order, including repeated core
    std::uint32_t cache_slots = 0;
};

// Stateless schedule only: does not load weights or implement inference.
[[nodiscard]] ExecutionPlan make_execution_plan(const TextConfig& config);

} // namespace ninfer::models::inkling2

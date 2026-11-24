#include "models/inkling2/config.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {
namespace {
using artifact::ArtifactError;
using artifact::Json;

std::uint32_t uint_field(const Json& obj, const char* key, bool positive = true) {
    const auto n = artifact::require_u64(obj.at(key), key, positive);
    if (n > std::numeric_limits<std::uint32_t>::max())
        throw ArtifactError(std::string("Inkling2 ") + key + ": out of uint32 range");
    return static_cast<std::uint32_t>(n);
}

float positive_float(const Json& obj, const char* key) {
    const auto& value = obj.at(key);
    if (!value.is_number()) throw ArtifactError(std::string("Inkling2 ") + key + ": not a number");
    const float number = value.get<float>();
    if (!std::isfinite(number) || number <= 0)
        throw ArtifactError(std::string("Inkling2 ") + key + ": expected positive finite");
    return number;
}

void require_value(const Json& obj, const char* key, const Json& expected) {
    if (obj.at(key) != expected)
        throw ArtifactError(std::string("Inkling2 ") + key + ": unsupported checkpoint variant");
}

void require_bool(const Json& obj, const char* key, bool expected) {
    if (!obj.at(key).is_boolean() || obj.at(key).get<bool>() != expected)
        throw ArtifactError(std::string("Inkling2 ") + key + ": unsupported boolean setting");
}

void require_equal(const char* label, std::uint32_t actual, std::uint32_t expected) {
    if (actual != expected)
        throw ArtifactError(std::string("Inkling2 ") + label + ": unsupported geometry");
}
} // namespace

CheckpointContract parse_hf_config(const artifact::Json& root) {
    try {
        require_value(root, "model_type", "inkling2_loop");
        require_value(root, "architectures", Json::array({"Inkling2LoopForConditionalGeneration"}));
        require_value(root.at("quantization_config"), "quant_algo", "NVFP4");
        const auto& t = root.at("text_config");
        require_value(t, "model_type", "inkling2_loop_text");
        const auto& loop = t.at("loop_config");
        CheckpointContract out{};
        auto& c = out.text;
        c.hidden_size       = uint_field(t, "hidden_size");
        c.physical_layers   = uint_field(t, "num_hidden_layers");
        c.vocabulary_rows   = uint_field(t, "vocab_size");
        c.public_tokens     = uint_field(t, "unpadded_vocab_size");
        c.eos_token_id      = uint_field(root, "eos_token_id", false);
        c.attention_heads   = uint_field(t, "num_attention_heads");
        c.kv_heads          = uint_field(t, "num_key_value_heads");
        c.head_dim          = uint_field(t, "head_dim");
        c.experts           = uint_field(t, "n_routed_experts");
        c.top_k             = uint_field(t, "num_experts_per_tok");
        c.intermediate     = uint_field(t, "intermediate_size");
        c.sliding_window   = uint_field(t, "sliding_window_size");
        c.convolution_kernel = uint_field(t, "sconv_kernel_size");
        c.block_layers     = uint_field(t, "n_layers_per_attn_res_block");
        c.core_start       = uint_field(loop, "core_start", false);
        c.core_end         = uint_field(loop, "core_end");
        c.core_passes      = uint_field(loop, "num_passes");
        c.gate_rank        = uint_field(loop, "gate_rank");
        c.norm_epsilon     = positive_float(t, "rms_norm_eps");
        c.gate_softcap     = positive_float(t, "mlp_gate_softcap");
        c.up_softcap       = positive_float(t, "mlp_up_softcap");
        c.logit_softcap    = positive_float(t, "final_logit_softcapping");
        c.route_scale      = positive_float(t, "route_scale");
        require_bool(t, "use_gate_bias", true);
        c.gate_bias = true;
        require_bool(loop, "is_varloop", false);
        c.varloop = false;
        require_equal("hidden_size", c.hidden_size, 2304);
        require_equal("num_hidden_layers", c.physical_layers, 24);
        require_equal("vocab_size", c.vocabulary_rows, 201024);
        require_equal("unpadded_vocab_size", c.public_tokens, 200058);
        require_equal("num_attention_heads", c.attention_heads, 24);
        require_equal("num_key_value_heads", c.kv_heads, 8);
        require_equal("head_dim", c.head_dim, 128);
        require_equal("n_routed_experts", c.experts, 128);
        require_equal("num_experts_per_tok", c.top_k, 8);
        require_equal("intermediate_size", c.intermediate, 1152);
        require_equal("sliding_window_size", c.sliding_window, 1024);
        require_equal("sconv_kernel_size", c.convolution_kernel, 4);
        require_equal("n_layers_per_attn_res_block", c.block_layers, 6);
        require_equal("core_start", c.core_start, 6);
        require_equal("core_end", c.core_end, 18);
        require_equal("num_passes", c.core_passes, 3);
        require_equal("gate_rank", c.gate_rank, 64);
        if (c.public_tokens > c.vocabulary_rows || c.eos_token_id >= c.public_tokens)
            throw ArtifactError("Inkling2 public token domain is invalid");
        // The published Mini model has 20 sliding and four full-attention layers.
        const auto& locals = t.at("local_layer_ids");
        if (!locals.is_array() || locals.size() != 20)
            throw ArtifactError("Inkling2 expected 20 local layer IDs");
        for (const auto& value : locals) {
            const auto layer = artifact::require_u64(value, "Inkling2 local layer", false);
            if (layer >= c.physical_layers || c.sliding[layer])
                throw ArtifactError("Inkling2 local layer out of range or repeated");
            c.sliding[layer] = true;
        }
        for (std::uint32_t layer = 0; layer < c.physical_layers; ++layer) {
            if (c.sliding[layer] != (layer % 6 != 5))
                throw ArtifactError("Inkling2 local/full layer schedule differs from Mini v1");
        }
        require_equal("n_shared_experts", uint_field(t, "n_shared_experts", false), 0);
        require_equal("d_rel", uint_field(t, "d_rel"), 16);
        require_equal("rel_extent", uint_field(t, "rel_extent"), 1024);
        require_bool(t, "use_value_norm_no_gamma", true);
        out.vision_present = root.contains("vision_config");
        out.mtp_present = root.contains("mtp_config");
        (void)make_execution_plan(c); // validates cache-slot geometry too
        return out;
    } catch (const ArtifactError&) {
        throw;
    } catch (const std::exception& e) {
        throw ArtifactError(std::string("Inkling2 HF config: ") + e.what());
    }
}

} // namespace ninfer::models::inkling2

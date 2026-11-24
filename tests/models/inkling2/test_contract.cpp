#include "models/inkling2/config.h"
#include "models/inkling2/plan.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ninfer::models::inkling2::Stage;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Mutate>
void expect_invalid(const ninfer::artifact::Json& original, Mutate mutate) {
    auto changed = original;
    mutate(changed);
    try {
        (void)ninfer::models::inkling2::parse_hf_config(changed);
    } catch (const ninfer::artifact::ArtifactError&) {
        return;
    }
    throw std::runtime_error("invalid Inkling2 config accepted");
}
} // namespace

int main() {
    try {
        std::ifstream stream(std::string(NINFER_SOURCE_DIR) +
                             "/tests/models/inkling2/fixtures/config.json");
        require(stream.good(), "Inkling2 HF config test fixture missing");
        ninfer::artifact::Json json;
        stream >> json;
        const auto c = ninfer::models::inkling2::parse_hf_config(json);
        require(c.text.public_tokens == 200058 && c.text.vocabulary_rows == 201024,
                "Inkling2 physical/public vocab geometry");
        require(c.vision_present && c.mtp_present, "Inkling2 HF companion presence");
        const auto p = ninfer::models::inkling2::make_execution_plan(c.text);
        require(p.cache_slots == 48 && p.steps.size() == 48, "Inkling2 logical step count");
        std::vector<bool> seen(48, false);
        for (std::size_t i = 0; i < p.steps.size(); ++i) {
            const auto& s = p.steps[i];
            require(s.cache_slot < seen.size() && !seen[s.cache_slot], "duplicate cache slot");
            seen[s.cache_slot] = true;
            require(s.block_boundary == (s.physical_layer % 6 == 0), "block boundary");
            require(s.sliding_attention == (s.physical_layer % 6 != 5), "attention kind");
            if (i < 6) {
                require(s.stage == Stage::Prelude && s.physical_layer == i &&
                            s.cache_slot == i && !s.reads_previous_pass_messages,
                        "Prelude schedule");
            } else if (i < 42) {
                const auto pass = static_cast<unsigned>((i - 6) / 12);
                const auto layer = static_cast<unsigned>(6 + (i - 6) % 12);
                const auto expected_slot = pass == 0 ? layer : 24 + (pass - 1) * 12 + layer - 6;
                require(s.stage == Stage::Core && s.physical_layer == layer &&
                            s.core_pass == pass && s.cache_slot == expected_slot &&
                            s.reads_previous_pass_messages == (pass > 0) &&
                            s.reset_to_prelude == (pass > 0 && layer == 6),
                        "Core schedule/message/cache slot");
            } else {
                const auto layer = static_cast<unsigned>(18 + i - 42);
                require(s.stage == Stage::Postlude && s.physical_layer == layer &&
                            s.cache_slot == layer && !s.reads_previous_pass_messages,
                        "Postlude schedule");
            }
        }
        for (bool was_seen : seen) require(was_seen, "missing cache slot");
        expect_invalid(json, [](auto& j) { j["text_config"]["loop_config"]["num_passes"] = 2; });
        expect_invalid(json, [](auto& j) { j["text_config"]["local_layer_ids"][1] = 0; });
        expect_invalid(json, [](auto& j) { j["text_config"]["unpadded_vocab_size"] = 201025; });
        expect_invalid(json, [](auto& j) { j["text_config"]["n_routed_experts"] = 256; });
        std::cout << "OK Inkling2 Mini: config contract, 48-step schedule, 48 cache slots,"
                     " message gates, negative cases\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL Inkling2 Mini contract: " << e.what() << '\n';
        return 1;
    }
}

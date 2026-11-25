#include "models/inkling2/batch_dispatch_stage6f.h"
#include "models/inkling2/plan.h"
#include <array>
#include <cstdint>
#include <type_traits>

namespace {
constexpr std::uint32_t kVisits = 48;
static_assert(std::is_standard_layout_v<InklingBatchVisit6F>);
static_assert(sizeof(InklingBatchVisit6F) == 9*sizeof(std::uint32_t));

bool create(InklingBatchVisit6F* output, std::uint32_t capacity,
            std::uint32_t tokens) {
    if (!output || capacity<kVisits || tokens<1 || tokens>4096) return false;
    ninfer::models::inkling2::TextConfig config{};
    config.physical_layers=24;
    config.core_start=6;config.core_end=18;config.core_passes=3;config.block_layers=6;
    for(std::uint32_t l=0;l<24;++l) config.sliding[l] = (l%6)!=5;
    const auto plan=ninfer::models::inkling2::make_execution_plan(config);
    if(plan.cache_slots!=kVisits || plan.steps.size()!=kVisits) return false;
    for(std::uint32_t index=0;index<kVisits;++index) {
        const auto& s=plan.steps[index];
        output[index]={index,s.physical_layer,s.cache_slot,s.core_pass,
            static_cast<std::uint32_t>(s.stage),
            s.block_boundary?1u:0u,s.reads_previous_pass_messages?1u:0u,
            s.reset_to_prelude?1u:0u,tokens};
    }
    return true;
}
} // namespace

extern "C" int inkling6f_plan(InklingBatchVisit6F* visits,
                               std::uint32_t capacity,
                               std::uint32_t tokens) noexcept {
    try {return create(visits,capacity,tokens)?static_cast<int>(kVisits):-1;}
    catch (...) {return -2;}
}
extern "C" int inkling6f_dispatch(std::uint32_t tokens,
                                   InklingVisitCallback6F callback,
                                   void* context) noexcept {
    if (!callback) return -1;
    try {
        std::array<InklingBatchVisit6F,kVisits> visits{};
        if(!create(visits.data(),kVisits,tokens))return -1;
        for (const auto& item:visits) {
            const int rc=callback(&item,context);
            if(rc!=0)return rc>0?rc:-3;
        }
        return static_cast<int>(kVisits);
    } catch (...) {return -2;}
}

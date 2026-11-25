#include "models/inkling2/batch_dispatch_stage6f.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

struct Recorder {
    std::vector<InklingBatchVisit6F> observed;
    unsigned total_tokens=0, resets=0, messages=0;
};
extern "C" int record(const InklingBatchVisit6F* step,void* opaque) {
    auto* r=static_cast<Recorder*>(opaque);
    r->observed.push_back(*step);
    r->total_tokens+=step->tokens;
    r->resets+=step->reset_to_prelude;
    r->messages+=step->reads_messages;
    return 0;
}
extern "C" int abort_early(const InklingBatchVisit6F* step,void*) {
    return step->ordinal==5? 17:0;
}
int main() {
    std::array<InklingBatchVisit6F,48> steps{};
    assert(inkling6f_plan(steps.data(),47,16)<0);
    assert(inkling6f_plan(steps.data(),48,0)<0);
    assert(inkling6f_plan(nullptr,48,16)<0);
    assert(inkling6f_plan(steps.data(),48,16)==48);
    std::set<int> slots;
    unsigned resets=0,messages=0,full=0;
    for(unsigned i=0;i<48;++i) {
        const auto& v=steps[i];
        assert(v.ordinal==i&&v.tokens==16&&v.physical_layer<24&&v.cache_slot<48);
        assert(slots.insert(v.cache_slot).second);
        if(v.physical_layer%6==5) ++full;
        resets+=v.reset_to_prelude; messages+=v.reads_messages;
    }
    assert(full==8); // layer 5 + 3 times layers 11,17 + layer 23
    assert(resets==2 && messages==24 && slots.size()==48);
    assert(steps[0].physical_layer==0 && steps[5].physical_layer==5);
    assert(steps[6].physical_layer==6 && steps[17].physical_layer==17);
    assert(steps[18].physical_layer==6 && steps[18].cache_slot==24 && steps[18].reset_to_prelude);
    assert(steps[29].cache_slot==35);
    assert(steps[30].physical_layer==6 && steps[30].cache_slot==36 && steps[30].reset_to_prelude);
    assert(steps[41].cache_slot==47);
    assert(steps[42].physical_layer==18 && steps[42].cache_slot==18);
    assert(steps[47].physical_layer==23 && steps[47].cache_slot==23);
    Recorder rec;
    assert(inkling6f_dispatch(16,record,&rec)==48);
    assert(rec.observed.size()==48 && rec.total_tokens==768 && rec.resets==2 && rec.messages==24);
    assert(inkling6f_dispatch(16,abort_early,nullptr)==17);
    assert(inkling6f_dispatch(16,nullptr,nullptr)<0);
    std::cout<<"OK Stage6F: 48 native C++ batch visits, 48 cache slots, 2 resets, "
             <<"24 message visits, 16-token chunk; NO model inference"<<std::endl;
}

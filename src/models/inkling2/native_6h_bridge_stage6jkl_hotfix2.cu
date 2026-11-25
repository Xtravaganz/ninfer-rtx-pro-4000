// Stage 6J/K/L Hotfix 2 — ABI-only safety bridge.
//
// Stage 6G/H's original strong inkling6h_run implementation wins if present.
// This weak implementation fills in a missing C export in a locally divergent
// native_resident_stage6gh.cu. It invokes the exact Stage 6G layer view and
// Stage 6E CUDA kernel; no host routing, no synchronous MoE, no weight copies.
#include "models/inkling2/native_resident_stage6gh.h"
#include <cstdint>

extern "C" int inkling6e_run(void*, unsigned long long, unsigned long long,
    unsigned long long, float, int, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long) noexcept;

extern "C" __attribute__((weak, visibility("default")))
int inkling6h_run(void* store, void* workspace, int layer,
    std::uint64_t hidden, std::uint64_t router, std::uint64_t bias,
    float route_scale, std::uint64_t output, std::uint64_t stream) noexcept {
    std::uint64_t pointers = 0, factors = 0, valid = 0;
    int format = 0;
    const int status = inkling6g_layer_view(store, layer, &pointers, &factors, &valid, &format);
    if (status) return status;
    return inkling6e_run(workspace, hidden, router, bias, route_scale,
                         format, pointers, factors, valid, output, stream);
}

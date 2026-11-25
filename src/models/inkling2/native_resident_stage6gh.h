#pragma once
#include <cstdint>

// Stage 6G/H: lifetime-owned compressed Inkling expert banks. All 128 experts of
// any uploaded physical layer remain resident until store_destroy. All GPU views
// are immutable. No GPU->Host top-k readback, no per-call allocation or eviction.
extern "C" {
void* inkling6g_store_create() noexcept;
void inkling6g_store_destroy(void*) noexcept;
// Host source-order concatenations: [128, rows, columns] per projection,
// NVFP4 nibble-packed codes + unswizzled G16 scales or MXFP8 bytes + G32 scales.
// factors is a Host float[4][128]: w13 amax, w13 denominator, w2 amax, w2 denominator.
// Upload is synchronous and safe for temporary/paged Host buffers; must run before
// consuming the layer. Returns 0 or CUDA error, 1001 for invalid arguments or 1002
// for already uploaded layer; failed partial uploads are cleaned up.
int inkling6g_upload_layer(void* store, int layer,
    std::uint64_t w13_codes, std::uint64_t w13_scales,
    std::uint64_t w2_codes, std::uint64_t w2_scales,
    std::uint64_t factors) noexcept;
int inkling6g_layer_view(void* store, int layer, std::uint64_t* pointers,
    std::uint64_t* factors, std::uint64_t* valid, int* format) noexcept;
std::uint64_t inkling6g_resident_bytes(void* store) noexcept;
int inkling6g_resident_layers(void* store) noexcept;
// Stage6H, one token. Workspaces belong to the prebuilt Stage6E CUDA code.
// Asynchronous: caller MUST keep x/router/bias/out and store alive on the same
// CUDA stream until all queued work completes; NO per-MoE cudaStreamSynchronize.
int inkling6h_run(void* store, void* workspace, int layer,
    std::uint64_t x, std::uint64_t router, std::uint64_t bias, float route_scale,
    std::uint64_t output, std::uint64_t stream) noexcept;
}

#pragma once
// Inkling 2 Mini Stage 5C: a streaming convolution cache keyed by the *logical*
// execution slot. This standalone prototype is NOT attached to ninfer-serve.
//
// Cache holds 48 logical visits x 4 convolution streams per visit; each stream
// has a separate input history. Core physical layers 6..17 are reused across
// 3 passes but their state slots 6..17, 24..35, 36..47 never alias.
//
// All tensor pointers are device pointers. Run/reset/create must use ONE CUDA
// stream, or the caller must explicitly synchronize cross-stream accesses.
// Never alias input x and output y, and do not mutate inputs before the stream
// has finished. Every call queues both the output and history update.
#include "models/inkling2/forward_primitives.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>

namespace ninfer::models::inkling2 {

// -1 indicates an invalid physical_layer/core_pass pair; no exceptions.
int inkling2_cache_slot(int physical_layer, int core_pass) noexcept;

enum class ConvMode5C : int { static_causal = 0, dynamic_kv = 1 };

class ConvCache48 {
public:
    ConvCache48() noexcept = default;
    ConvCache48(const ConvCache48&) = delete;
    ConvCache48& operator=(const ConvCache48&) = delete;
    ~ConvCache48() noexcept;

    // Allocates two history buffers for each of 48 slots x 4 convolution
    // streams. History is FP32 [B,K-1,max_channels], older samples first.
    // Invalid source IDs are -1. Resets are asynchronous on provided stream.
    cudaError_t create(int batch, int max_channels, int kernel,
                       cudaStream_t stream = nullptr) noexcept;
    cudaError_t reset(cudaStream_t stream = nullptr) noexcept;
    void release() noexcept;

    // x/y: [B,T,C], FP32 or BF16. static_weight: [C,K] FP32:
    //  - static mode: oldest tap first, /(K+1)^0.5;
    //  - dynamic mode: lag 0 (current) first, /(2K+1)^0.5;
    // dynamic_taps in dynamic mode: [B,T,H,K] FP32, already projected from
    // original hidden input; C must be divisible by H. For static mode it must
    // be nullptr, num_heads must be 0.
    // mask: [B,T] byte, optional; only convolution input is masked.
    // seq_ids: [B,T] int32, optional in dynamic mode (otherwise ignored).
    // On dynamic mode, historical tokens with a different sequence ID are
    // excluded; a cleared cache carries sequence ID -1 and input value zero.
    // conv_idx: K=0, V=1, post_attn=2, post_mlp=3.
    // Sliding layers (physical layer % 6 != 5) use STATIC K/V convolutions.
    // Full-attention layers (5,11,17,23) use DYNAMIC K/V convolutions.
    cudaError_t run(int physical_layer, int core_pass, int conv_idx,
                    ConvMode5C mode, const void* x,
                    const float* static_weight, const float* dynamic_taps,
                    const std::uint8_t* padding_mask,
                    const std::int32_t* seq_ids, void* y,
                    int sequence, int channels, int num_heads,
                    ForwardScalar dtype, cudaStream_t stream = nullptr) noexcept;

    bool ready() const noexcept { return history_ != nullptr; }
    int batch() const noexcept { return batch_; }
    int kernel() const noexcept { return kernel_; }

private:
    static constexpr int kSlots = 48;
    static constexpr int kStreams = 4;
    static constexpr int kCopies = 2;
    static constexpr int kGroups = kSlots * kStreams * kCopies;
    float* history_ = nullptr;
    std::int32_t* ids_ = nullptr;
    int batch_ = 0;
    int max_channels_ = 0;
    int kernel_ = 0;
    std::uint8_t parity_[kSlots * kStreams]{};
    int bound_channels_[kSlots * kStreams]{};
    int bound_heads_[kSlots * kStreams]{};
    int bound_mode_[kSlots * kStreams]{};
    int bound_dtype_[kSlots * kStreams]{};

    std::size_t group_index(int slot, int conv_idx, int copy) const noexcept;
    float* history_group(std::size_t group) const noexcept;
    std::int32_t* id_group(std::size_t group) const noexcept;
};

}  // namespace ninfer::models::inkling2

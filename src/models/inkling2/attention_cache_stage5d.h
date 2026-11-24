#pragma once
// Stage 5D: standalone device-resident attention K/V cache + prefill/decode
// correctness backend. Physical core layers share weights but NEVER K/V slots.
// Not registered in the NInfer runtime; caller provides already projected Q/K/V,
// clipped/scaled per-query relative states and sink logits.
#include "models/inkling2/forward_primitives.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>

namespace ninfer::models::inkling2 {

class AttentionKv48 {
public:
    AttentionKv48() noexcept = default;
    AttentionKv48(const AttentionKv48&) = delete;
    AttentionKv48& operator=(const AttentionKv48&) = delete;
    ~AttentionKv48() noexcept;

    // dtype is fixed for the cache. Allocates 48 separate logical slots.
    // Caller must use one CUDA stream (or synchronize across streams).
    cudaError_t create(int batch, int kv_heads, int dim, int capacity,
                       ForwardScalar dtype, cudaStream_t stream = nullptr) noexcept;
    cudaError_t reset(cudaStream_t stream = nullptr) noexcept;
    void release() noexcept;
    bool ready() const noexcept { return keys_ != nullptr; }
    int size(int physical_layer, int core_pass) const noexcept;

    // All pointers device resident. Q [B,T,Hq,D], K/V [B,T,Hkv,D], Y [B,T,Hq,D].
    // relative_states: optional [B,T,Hq,Rdim] FP32, pre-normalized / clipped,
    //   bias_bank [Rdim, rel_extent] FP32; if Rdim=0, both must be null.
    // sink_logits [B,T,Hq] FP32; key_valid [B,T] uint8 (optional),
    // seq_ids [B,T] int32 (optional). Key masking includes causal, sliding,
    // valid byte and sequence-id equality. Sink absorbs softmax mass with value 0.
    // sliding_window=0 means full attention, positive otherwise.
    // This correctness-first kernel is O(T * context) and not optimized prefill.
    cudaError_t append_and_attend(
        int physical_layer, int core_pass,
        const void* q, const void* k, const void* v,
        const float* relative_states, const float* bias_bank,
        const float* sink_logits, const std::uint8_t* key_valid,
        const std::int32_t* seq_ids, void* y,
        int tokens, int q_heads, int rel_dim, int rel_extent,
        int sliding_window, cudaStream_t stream = nullptr) noexcept;

private:
    void* keys_ = nullptr;
    void* values_ = nullptr;
    std::uint8_t* valid_ = nullptr;
    std::int32_t* seq_ = nullptr;
    int lengths_[48]{};
    int head_count_[48]{};
    int batch_ = 0, kv_heads_ = 0, dim_ = 0, capacity_ = 0;
    ForwardScalar dtype_ = ForwardScalar::float32;
    std::size_t total_words_ = 0;
};

} // namespace ninfer::models::inkling2

#pragma once
// Stage 5E: correctness-first, single-token FP32 Top-8 Sparse-MoE GPU path.
// Device-resident source-order NVFP4 E2M1/G16/E4M3 or MXFP8 E4M3/G32/UE8M0.
// No dense FP32 weight materialization, no ninfer-serve registry integration.
// Real-checkpoint bank upload, router state and asynchronous residency scheduling
// remain caller responsibilities. One CUDA stream per workspace.
#include <cuda_runtime_api.h>
#include <cstdint>

namespace ninfer::models::inkling2 {
constexpr int kInklingHidden5E=2304;
constexpr int kInklingIntermediate5E=1152;
constexpr int kInklingExperts5E=128;
constexpr int kInklingTopK5E=8;

enum class ExpertFormat5E:int { nvfp4_e2m1_g16=1, mxfp8_e4m3_g32=2 };
struct ExpertBank5E {
    // codes/scales are [resident,N,K/2|K] and [resident,N,K/16|K/32]
    // in NATURAL SOURCE ROW-MAJOR order (Stage2A .i2bank records).
    const std::uint8_t* codes=nullptr;
    const std::uint8_t* scales=nullptr;
    // Only for NVFP4: one valid positive FP32 word each per resident expert.
    const float* amax=nullptr;
    const float* denominator=nullptr;
    ExpertFormat5E format=ExpertFormat5E::nvfp4_e2m1_g16;
    int resident=0;
    int rows=0,cols=0;
};

// Performs HF-style routing: sigmoid(logits), correction bias for selection
// ONLY, unadjusted scores for normalized weights, *route_scale afterwards.
// Output top_ids[8] int32 + weights[8] FP32, on device.
// Deterministic tie-breaking to smaller expert ID for testing.
cudaError_t launch_top8_router_5e(
    const float* hidden, const float* router_weight,
    const float* correction_bias, float route_scale,
    std::int32_t* top_ids, float* top_weights,
    cudaStream_t stream=nullptr) noexcept;

class MoEWorkspace5E {
public:
    MoEWorkspace5E() noexcept=default;
    MoEWorkspace5E(const MoEWorkspace5E&)=delete;
    MoEWorkspace5E& operator=(const MoEWorkspace5E&)=delete;
    ~MoEWorkspace5E() noexcept;
    cudaError_t create() noexcept;
    void release() noexcept;
    bool ready() const noexcept { return ids_!=nullptr; }
    const std::int32_t* top_ids() const noexcept { return ids_; }
    const float* top_weights() const noexcept { return weights_; }
    // Missing selected experts: missing_resident_flag() becomes 1, output
    // NaN. Caller MUST inspect flag after stream synchronization.
    const int* missing_resident_flag() const noexcept { return missing_; }

    // map[128] int32 maps physical expert ID to resident index, -1 if absent.
    // W13 [2304,2304]; W2 [2304,1152], same quantization per physical layer.
    // After routing and selected experts are resident, run the 2 GEMVs +
    // HF softcapped SwiGLU + weighted reduction for the single input token.
    cudaError_t selected(const float* hidden, const ExpertBank5E& w13,
                         const ExpertBank5E& w2, const std::int32_t* resident_map,
                         float* out, cudaStream_t stream=nullptr) noexcept;
    // Convenience for fully resident or pre-staged top8: router then selected.
    cudaError_t run(const float* hidden,const float* router_weight,
                    const float* correction_bias,float route_scale,
                    const ExpertBank5E& w13,const ExpertBank5E& w2,
                    const std::int32_t* resident_map,float* out,
                    cudaStream_t stream=nullptr) noexcept;
private:
    std::int32_t* ids_=nullptr;
    float *weights_=nullptr,*up_=nullptr,*activation_=nullptr,*down_=nullptr;
    int* missing_=nullptr;
};
} // namespace ninfer::models::inkling2

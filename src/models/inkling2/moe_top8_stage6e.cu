#include "models/inkling2/moe_top8_stage5e.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace ninfer::models::inkling2 {
namespace {
constexpr int H=kInklingHidden5E, I=kInklingIntermediate5E, E=kInklingExperts5E, K=kInklingTopK5E;
__device__ __forceinline__ float e4m3(std::uint8_t w) {
    if ((w&0x7fu)==0x7fu)return __uint_as_float(0x7fc00000u);
    const unsigned sign=static_cast<unsigned>(w&0x80u)<<24;
    const unsigned ex=(w>>3)&15u,man=w&7u;
    if(ex==0u)return __uint_as_float(__float_as_uint(float(man)*0x1p-9f)|sign);
    return __uint_as_float(sign|((ex+120u)<<23)|(man<<20));
}
__device__ __forceinline__ float ue8m0(std::uint8_t b) {
    if(b==0xffu)return __uint_as_float(0x7fc00000u);
    return __uint_as_float(b==0u?0x00400000u:(static_cast<unsigned>(b)<<23));
}
__device__ __forceinline__ float e2m1(std::uint8_t n) {
    constexpr float magnitudes[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    const float value=magnitudes[n&7u];
    return (n&8u)?-value:value;
}
__global__ void router_kernel(const float* x,const float* W,const float* bias,
                               float scale,std::int32_t* ids,float* weights) {
    const int e=static_cast<int>(threadIdx.x);
    __shared__ float score[E],rank_score[E];
    float dot=0.f;
    for(int i=0;i<H;++i)dot=fmaf(x[i],W[static_cast<std::size_t>(e)*H+i],dot);
    const float s=1.f/(1.f+expf(-dot));
    score[e]=s;rank_score[e]=s+bias[e];
    __syncthreads();
    if(e==0){
        float total=0.f;
        for(int r=0;r<K;++r){
            int best=-1;
            float maximum=-CUDART_INF_F;
            for(int j=0;j<E;++j)if(rank_score[j]>maximum){maximum=rank_score[j];best=j;}
            ids[r]=best;
            weights[r]=score[best];total+=score[best];
            rank_score[best]=-CUDART_INF_F;
        }
        const float multiplier=scale/(total+1e-20f);
        for(int r=0;r<K;++r)weights[r]*=multiplier;
    }
}
// Stage6E: four uint64_t pointer tables [w13 codes, w13 scales, w2 codes, w2 scales]
// indexed by the PHYSICAL expert ID (0..127). Each pointer owns an existing
// GPU-resident source-ordered, unswizzled quantized expert tensor. NO GPU gather.
__global__ void resident_kernel_6e(const std::int32_t* ids,
                                 const std::uint64_t* ptrs,
                                 const std::int32_t* valid,
                                 int* missing) {
    if (threadIdx.x == 0) {
        int bad = 0;
        for (int r = 0; r < K; ++r) {
            const int id = ids[r];
            if (id < 0 || id >= E || valid[id] != id ||
                !ptrs[id] || !ptrs[E+id] || !ptrs[2*E+id] || !ptrs[3*E+id]) bad = 1;
        }
        *missing = bad;
    }
}

template<int FMT, bool PerRouteInput> __global__ void gemv_direct_6e(
    const std::uint64_t* ptrs, const float* factors,
    const float* x, const std::int32_t* selected,
    const std::int32_t* valid, float* out, int rows, int cols, int plane) {
    const int row = static_cast<int>(blockIdx.x), route = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int eid = selected[route];
    if (eid < 0 || eid >= E || valid[eid] != eid ||
        !ptrs[plane*2*E + eid] || !ptrs[(plane*2+1)*E + eid]) {
        if (lane == 0) out[static_cast<std::size_t>(route)*rows+row] = CUDART_NAN_F;
        return;
    }
    const auto* codes = reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(ptrs[plane*2*E+eid]));
    const auto* scales = reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(ptrs[(plane*2+1)*E+eid]));
    const int group_size = FMT == 1 ? 16 : 32;
    const float factor = FMT == 1 ?
        factors[plane*2*E+eid] / factors[(plane*2+1)*E+eid] : 1.f;
    float sum = 0.f;
    for (int col=lane; col<cols; col+=128) {
        const std::size_t base = static_cast<std::size_t>(row)*cols+col;
        float w;
        if constexpr (FMT==1) {
            const std::uint8_t packed = codes[base/2];
            const std::uint8_t nibble = (col & 1) ? packed >> 4 : packed & 15;
            w = e2m1(nibble) * e4m3(scales[static_cast<std::size_t>(row)*(cols/16)+col/16]) * factor;
        } else {
            w = e4m3(codes[base]) * ue8m0(scales[static_cast<std::size_t>(row)*(cols/32)+col/32]);
        }
        sum = fmaf(w, x[(PerRouteInput ? static_cast<std::size_t>(route)*cols : 0)+col], sum);
    }
    __shared__ float tmp[128]; tmp[lane]=sum;
    __syncthreads();
    for (int offset=64; offset>=1; offset>>=1) {
        if (lane<offset) tmp[lane] += tmp[lane+offset];
        __syncthreads();
    }
    if (lane==0) out[static_cast<std::size_t>(route)*rows+row] = tmp[0];
}
__global__ void swiglu_kernel(const float* up,float* act) {
    const int i=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=K*I)return;
    const int route=i/I,neuron=i%I;
    const float gate=up[route*2*I+2*neuron];
    const float value=up[route*2*I+2*neuron+1];
    const float g=4.f*tanhf(gate/4.f)/(1.f+expf(-gate));
    const float u=25.f*tanhf(value/25.f);
    act[i]=g*u;
}
__global__ void weighted_reduce(const float* down,const float* weights,
                                const int* missing,float* out) {
    const int row=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=H)return;
    if(*missing){out[row]=CUDART_NAN_F;return;}
    float sum=0.f;
    for(int i=0;i<K;++i)sum=fmaf(weights[i],down[i*H+row],sum);
    out[row]=sum;
}
inline bool valid(const ExpertBank5E& b,int rows,int cols) {
    const bool fmt=b.format==ExpertFormat5E::nvfp4_e2m1_g16||b.format==ExpertFormat5E::mxfp8_e4m3_g32;
    return fmt&&b.codes&&b.scales&&b.resident>0&&b.resident<=E&&b.rows==rows&&b.cols==cols&&
        (b.format!=ExpertFormat5E::nvfp4_e2m1_g16||(b.amax&&b.denominator));
}
}

// Stage6E uses its own workspace type, avoiding changes to Stage5E or Stage5W.
struct Workspace6E {
    std::int32_t* ids = nullptr;
    float *weights = nullptr, *up = nullptr, *activation = nullptr, *down = nullptr;
    int* missing = nullptr;
    ~Workspace6E() noexcept { release(); }
    void release() noexcept {
        if (ids) cudaFree(ids); if (weights) cudaFree(weights);
        if (up) cudaFree(up); if (activation) cudaFree(activation);
        if (down) cudaFree(down); if (missing) cudaFree(missing);
        ids=nullptr; weights=up=activation=down=nullptr; missing=nullptr;
    }
    cudaError_t create() noexcept {
        if (ids) return cudaErrorInvalidValue;
        cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&ids), K*sizeof(std::int32_t));
        if (e!=cudaSuccess) {release();return e;}
        e=cudaMalloc(reinterpret_cast<void**>(&weights),K*sizeof(float)); if(e!=cudaSuccess){release();return e;}
        e=cudaMalloc(reinterpret_cast<void**>(&up),K*2*I*sizeof(float)); if(e!=cudaSuccess){release();return e;}
        e=cudaMalloc(reinterpret_cast<void**>(&activation),K*I*sizeof(float)); if(e!=cudaSuccess){release();return e;}
        e=cudaMalloc(reinterpret_cast<void**>(&down),K*H*sizeof(float)); if(e!=cudaSuccess){release();return e;}
        e=cudaMalloc(reinterpret_cast<void**>(&missing),sizeof(int)); if(e!=cudaSuccess){release();return e;}
        return cudaSuccess;
    }
    cudaError_t run(const float* hidden,const float* router,const float* bias,
                    float route_scale,int fmt,const std::uint64_t* pointers,
                    const float* factors,const std::int32_t* valid,float* out,
                    cudaStream_t stream) noexcept {
        if (!ids || !hidden || !router || !bias || !pointers || !factors || !valid || !out ||
            hidden==out || (fmt!=1 && fmt!=2) || !std::isfinite(route_scale) || route_scale<=0)
            return cudaErrorInvalidValue;
        router_kernel<<<1,E,0,stream>>>(hidden,router,bias,route_scale,ids,weights);
        cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) return e;
        resident_kernel_6e<<<1,32,0,stream>>>(ids,pointers,valid,missing);
        e=cudaGetLastError(); if(e!=cudaSuccess) return e;
        const dim3 up_blocks(2*I,K), down_blocks(H,K);
        if(fmt==1) {
            gemv_direct_6e<1,false><<<up_blocks,128,0,stream>>>(pointers,factors,hidden,ids,valid,up,2*I,H,0);
        } else {
            gemv_direct_6e<2,false><<<up_blocks,128,0,stream>>>(pointers,factors,hidden,ids,valid,up,2*I,H,0);
        }
        e=cudaGetLastError(); if(e!=cudaSuccess) return e;
        swiglu_kernel<<<(K*I+255)/256,256,0,stream>>>(up,activation);
        e=cudaGetLastError(); if(e!=cudaSuccess) return e;
        if(fmt==1) {
            gemv_direct_6e<1,true><<<down_blocks,128,0,stream>>>(pointers,factors,activation,ids,valid,down,H,I,1);
        } else {
            gemv_direct_6e<2,true><<<down_blocks,128,0,stream>>>(pointers,factors,activation,ids,valid,down,H,I,1);
        }
        e=cudaGetLastError(); if(e!=cudaSuccess) return e;
        weighted_reduce<<<(H+255)/256,256,0,stream>>>(down,weights,missing,out);
        return cudaGetLastError();
    }
};
} // namespace ninfer::models::inkling2

extern "C" {
void* inkling6e_workspace_create() noexcept {
    auto* ptr=new(std::nothrow) ninfer::models::inkling2::Workspace6E();
    if(!ptr)return nullptr;
    if(ptr->create()!=cudaSuccess){delete ptr;return nullptr;}
    return ptr;
}
void inkling6e_workspace_destroy(void* p) noexcept {
    delete static_cast<ninfer::models::inkling2::Workspace6E*>(p);
}
int inkling6e_run(void* workspace, unsigned long long hidden, unsigned long long router,
    unsigned long long bias, float route_scale, int format,
    unsigned long long pointers, unsigned long long factors, unsigned long long valid,
    unsigned long long output, unsigned long long stream) noexcept {
    auto* w=static_cast<ninfer::models::inkling2::Workspace6E*>(workspace);
    if(!w)return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(w->run(
        reinterpret_cast<const float*>(hidden), reinterpret_cast<const float*>(router),
        reinterpret_cast<const float*>(bias), route_scale, format,
        reinterpret_cast<const std::uint64_t*>(pointers),
        reinterpret_cast<const float*>(factors),
        reinterpret_cast<const std::int32_t*>(valid),
        reinterpret_cast<float*>(output), reinterpret_cast<cudaStream_t>(stream)));
}
}

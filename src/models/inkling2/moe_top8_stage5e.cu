#include "models/inkling2/moe_top8_stage5e.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

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
__global__ void resident_kernel(const std::int32_t* ids,const std::int32_t* map,
                                int* missing,int resident) {
    if(threadIdx.x==0){
        int bad=0;
        for(int r=0;r<K;++r)if(ids[r]<0||ids[r]>=E||(map[ids[r]]<0||map[ids[r]]>=resident))bad=1;
        *missing=bad;
    }
}

template<int FMT, bool PerRouteInput> __global__ void gemv_kernel(
    const std::uint8_t* codes,const std::uint8_t* scales,
    const float* amax,const float* denominator,
    const float* x,const std::int32_t* selected,const std::int32_t* mapping,
    float* out,int rows,int cols,int resident) {
    const int row=static_cast<int>(blockIdx.x),route=static_cast<int>(blockIdx.y);
    const int lane=static_cast<int>(threadIdx.x);
    const int eid=selected[route];
    const int expert=mapping[eid];
    if(expert<0||expert>=resident){if(lane==0)out[static_cast<std::size_t>(route)*rows+row]=CUDART_NAN_F;return;}
    const std::size_t weight_start=static_cast<std::size_t>(expert)*rows*cols;
    const int group_size=FMT==1?16:32;
    const std::size_t scale_start=static_cast<std::size_t>(expert)*rows*(cols/group_size);
    const float ratio=FMT==1?amax[expert]/denominator[expert]:1.f;
    float sum=0.f;
    for(int col=lane;col<cols;col+=128){
        const std::size_t base=static_cast<std::size_t>(row)*cols+col;
        float w;
        if constexpr(FMT==1){
            const std::uint8_t packed=codes[(weight_start+base)/2];
            const std::uint8_t nibble=(col&1)?packed>>4:packed&0xf;
            w=e2m1(nibble)*e4m3(scales[scale_start+static_cast<std::size_t>(row)*(cols/16)+col/16])*ratio;
        }else{
            w=e4m3(codes[weight_start+base])*ue8m0(scales[scale_start+static_cast<std::size_t>(row)*(cols/32)+col/32]);
        }
        sum=fmaf(w,x[(PerRouteInput?static_cast<std::size_t>(route)*cols:0)+col],sum);
    }
    __shared__ float tmp[128];tmp[lane]=sum;
    __syncthreads();
    for(int offset=64;offset>=1;offset>>=1){
        if(lane<offset)tmp[lane]+=tmp[lane+offset];
        __syncthreads();
    }
    if(lane==0)out[static_cast<std::size_t>(route)*rows+row]=tmp[0];
}
__global__ void swiglu_kernel(const float* up,float* act) {
    const int i=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=K*I)return;
    const int route=i/I,neuron=i%I;
    const float gate=up[route*2*I+neuron];
    const float value=up[route*2*I+I+neuron];
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

cudaError_t launch_top8_router_5e(const float* hidden,const float* router_weight,
    const float* correction_bias,float route_scale,std::int32_t* top_ids,
    float* top_weights,cudaStream_t stream) noexcept {
    if(!hidden||!router_weight||!correction_bias||!top_ids||!top_weights||
       !std::isfinite(route_scale)||route_scale<=0.f)return cudaErrorInvalidValue;
    router_kernel<<<1,E,0,stream>>>(hidden,router_weight,correction_bias,route_scale,top_ids,top_weights);
    return cudaGetLastError();
}

MoEWorkspace5E::~MoEWorkspace5E() noexcept {release();}
void MoEWorkspace5E::release() noexcept {
    if(ids_)cudaFree(ids_);
    if(weights_)cudaFree(weights_);
    if(up_)cudaFree(up_);
    if(activation_)cudaFree(activation_);
    if(down_)cudaFree(down_);
    if(missing_)cudaFree(missing_);
    ids_=nullptr;weights_=up_=activation_=down_=nullptr;missing_=nullptr;
}
cudaError_t MoEWorkspace5E::create() noexcept {
    if(ready())return cudaErrorInvalidValue;
    cudaError_t e=cudaMalloc(reinterpret_cast<void**>(&ids_),K*sizeof(std::int32_t));
    if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&weights_),K*sizeof(float));if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&up_),K*2*I*sizeof(float));if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&activation_),K*I*sizeof(float));if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&down_),K*H*sizeof(float));if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&missing_),sizeof(int));if(e!=cudaSuccess){release();return e;}
    return cudaSuccess;
}
cudaError_t MoEWorkspace5E::selected(const float* hidden,const ExpertBank5E& w13,
    const ExpertBank5E& w2,const std::int32_t* resident_map,
    float* out,cudaStream_t stream) noexcept {
    if(!ready()||!hidden||!resident_map||!out||hidden==out||
       !valid(w13,2*I,H)||!valid(w2,H,I)||w13.format!=w2.format||
       w13.resident!=w2.resident)return cudaErrorInvalidValue;
    resident_kernel<<<1,32,0,stream>>>(ids_,resident_map,missing_,w13.resident);
    cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
    const dim3 blocks_up(2*I,K),blocks_down(H,K);
    if(w13.format==ExpertFormat5E::nvfp4_e2m1_g16) {
        gemv_kernel<1,false><<<blocks_up,128,0,stream>>>(w13.codes,w13.scales,w13.amax,w13.denominator,hidden,ids_,resident_map,up_,2*I,H,w13.resident);
    }else{
        gemv_kernel<2,false><<<blocks_up,128,0,stream>>>(w13.codes,w13.scales,nullptr,nullptr,hidden,ids_,resident_map,up_,2*I,H,w13.resident);
    }
    e=cudaGetLastError();if(e!=cudaSuccess)return e;
    swiglu_kernel<<<(K*I+255)/256,256,0,stream>>>(up_,activation_);
    e=cudaGetLastError();if(e!=cudaSuccess)return e;
    if(w2.format==ExpertFormat5E::nvfp4_e2m1_g16){
        gemv_kernel<1,true><<<blocks_down,128,0,stream>>>(w2.codes,w2.scales,w2.amax,w2.denominator,activation_,ids_,resident_map,down_,H,I,w2.resident);
    }else{
        gemv_kernel<2,true><<<blocks_down,128,0,stream>>>(w2.codes,w2.scales,nullptr,nullptr,activation_,ids_,resident_map,down_,H,I,w2.resident);
    }
    e=cudaGetLastError();if(e!=cudaSuccess)return e;
    weighted_reduce<<<(H+255)/256,256,0,stream>>>(down_,weights_,missing_,out);
    return cudaGetLastError();
}
cudaError_t MoEWorkspace5E::run(const float* hidden,const float* router_weight,
    const float* correction_bias,float route_scale,const ExpertBank5E& w13,
    const ExpertBank5E& w2,const std::int32_t* resident_map,
    float* out,cudaStream_t stream) noexcept {
    if(!ready())return cudaErrorInvalidValue;
    cudaError_t e=launch_top8_router_5e(hidden,router_weight,correction_bias,route_scale,ids_,weights_,stream);
    if(e!=cudaSuccess)return e;
    return selected(hidden,w13,w2,resident_map,out,stream);
}
} // namespace ninfer::models::inkling2

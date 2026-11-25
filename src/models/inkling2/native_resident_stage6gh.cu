#include "models/inkling2/native_resident_stage6gh.h"
#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <new>

extern "C" int inkling6e_run(void*, unsigned long long, unsigned long long,
    unsigned long long, float, int, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long) noexcept;
namespace {
constexpr int kLayers=24, kExperts=128, kHidden=2304, kIntermediate=1152;
struct Bank {
    std::array<std::uint8_t*,4> data{};
    std::uint64_t* pointers=nullptr;
    float* factors=nullptr;
    std::int32_t* valid=nullptr;
    std::uint64_t bytes=0;
    int format=0;
    bool ready=false;
    void clear() noexcept {
        for(auto& p:data){if(p) cudaFree(p);p=nullptr;}
        if(pointers)cudaFree(pointers); pointers=nullptr;
        if(factors)cudaFree(factors); factors=nullptr;
        if(valid)cudaFree(valid); valid=nullptr;
        ready=false;bytes=0;format=0;
    }
};
struct Store {
    std::array<Bank,kLayers> banks{};
    int device=-1;
    std::mutex lock;
    Store() {cudaGetDevice(&device);}
    ~Store() noexcept {cudaSetDevice(device);cudaDeviceSynchronize();for(auto& b:banks)b.clear();}
};
// Contract: Stage6E expects exact [128, rows, cols] source-order tensors.
std::size_t plane_bytes(int layer,int plane) {
    const bool nv=layer<20;
    const std::size_t cells=(plane<2 ? std::size_t(2*kIntermediate)*kHidden :
                                     std::size_t(kHidden)*kIntermediate);
    return std::size_t(kExperts)*cells / (plane%2 == 0 ? (nv?2:1) : (nv?16:32));
}
int upload(Store* store,int layer,const std::uint64_t* host,
           const float* factor_words) noexcept {
    if(!store||layer<0||layer>=kLayers||!factor_words)return 1001;
    for(int i=0;i<4;++i)if(!host[i])return 1001;
    // Upload is deliberately non-capturable: no resident mutation while kernels use it.
    std::lock_guard<std::mutex> guard(store->lock);
    if(store->banks[layer].ready)return 1002;
    cudaError_t e=cudaSetDevice(store->device);
    if(e!=cudaSuccess)return int(e);
    Bank fresh;
    std::array<std::uint64_t,4*kExperts> ptrs{};
    std::array<std::int32_t,kExperts> ids{};
    for(int i=0;i<kExperts;++i) {
        ids[i]=i;
        if(layer<20)
            for(int j=0;j<4;++j)
                if(!std::isfinite(factor_words[j*kExperts+i])||factor_words[j*kExperts+i]<=0)
                    return 1001;
    }
    for(int p=0;p<4;++p) {
        const auto n=plane_bytes(layer,p);
        const auto stride=n/kExperts;
        e=cudaMalloc(reinterpret_cast<void**>(&fresh.data[p]),n);
        if(e!=cudaSuccess){fresh.clear();return int(e);}
        e=cudaMemcpy(fresh.data[p],reinterpret_cast<const void*>(host[p]),n,cudaMemcpyHostToDevice);
        if(e!=cudaSuccess){fresh.clear();return int(e);}
        for(int id=0;id<kExperts;++id)
            ptrs[p*kExperts+id]=reinterpret_cast<std::uint64_t>(fresh.data[p]+id*stride);
        fresh.bytes+=n;
    }
    e=cudaMalloc(reinterpret_cast<void**>(&fresh.pointers),sizeof(ptrs));
    if(e!=cudaSuccess){fresh.clear();return int(e);}
    e=cudaMemcpy(fresh.pointers,ptrs.data(),sizeof(ptrs),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fresh.clear();return int(e);}
    e=cudaMalloc(reinterpret_cast<void**>(&fresh.factors),sizeof(float)*4*kExperts);
    if(e!=cudaSuccess){fresh.clear();return int(e);}
    e=cudaMemcpy(fresh.factors,factor_words,sizeof(float)*4*kExperts,cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fresh.clear();return int(e);}
    e=cudaMalloc(reinterpret_cast<void**>(&fresh.valid),sizeof(ids));
    if(e!=cudaSuccess){fresh.clear();return int(e);}
    e=cudaMemcpy(fresh.valid,ids.data(),sizeof(ids),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){fresh.clear();return int(e);}
    fresh.bytes+=sizeof(ptrs)+sizeof(float)*4*kExperts+sizeof(ids);
    fresh.format=layer<20?1:2;
    fresh.ready=true;
    store->banks[layer]=fresh;
    // fresh has no destructor, bank owns its device memory from here.
    return 0;
}
} // namespace
extern "C" {
void* inkling6g_store_create() noexcept {
    int dev=-1;if(cudaGetDevice(&dev)!=cudaSuccess)return nullptr;
    return new(std::nothrow) Store();
}
void inkling6g_store_destroy(void* p) noexcept {delete static_cast<Store*>(p);}
int inkling6g_upload_layer(void* p,int layer,std::uint64_t c13,std::uint64_t s13,
  std::uint64_t c2,std::uint64_t s2,std::uint64_t factors) noexcept {
    const std::uint64_t v[]{c13,s13,c2,s2};
    return upload(static_cast<Store*>(p),layer,v,reinterpret_cast<const float*>(factors));
}
int inkling6g_layer_view(void* p,int layer,std::uint64_t* ptrs,
   std::uint64_t* factors,std::uint64_t* valid,int* format) noexcept {
    auto* s=static_cast<Store*>(p);
    if(!s||layer<0||layer>=kLayers||!ptrs||!factors||!valid||!format)return 1001;
    const auto& b=s->banks[layer];if(!b.ready)return 1003;
    *ptrs=reinterpret_cast<std::uint64_t>(b.pointers);
    *factors=reinterpret_cast<std::uint64_t>(b.factors);
    *valid=reinterpret_cast<std::uint64_t>(b.valid);
    *format=b.format;return 0;
}
std::uint64_t inkling6g_resident_bytes(void* p) noexcept {
    auto* s=static_cast<Store*>(p);if(!s)return 0;
    std::uint64_t n=0;for(const auto& b:s->banks)n+=b.bytes;return n;
}
int inkling6g_resident_layers(void* p) noexcept {
    auto* s=static_cast<Store*>(p);if(!s)return 0;
    int n=0;for(const auto& b:s->banks)n+=int(b.ready);return n;
}
int inkling6h_run(void* p,void* w,int layer,std::uint64_t x,std::uint64_t r,
 std::uint64_t bias,float scale,std::uint64_t out,std::uint64_t stream) noexcept {
    std::uint64_t ptrs=0,factors=0,valid=0;int format=0;
    const int e=inkling6g_layer_view(p,layer,&ptrs,&factors,&valid,&format);
    if(e)return e;
    return inkling6e_run(w,x,r,bias,scale,format,ptrs,factors,valid,out,stream);
}
}

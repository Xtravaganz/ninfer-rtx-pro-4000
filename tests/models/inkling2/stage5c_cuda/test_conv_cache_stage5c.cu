// Inkling 2 Mini Stage 5C: CUDA stateful convolution correctness and parity.
// Independent scalar double oracle, compare every element (not global max only).
#include "models/inkling2/conv_cache_stage5c.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ink = ninfer::models::inkling2;
namespace {
void require(bool condition, const std::string& explanation) {
    if (!condition) throw std::runtime_error(explanation);
}
void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
template<class T> struct Device {
    T* ptr = nullptr;
    explicit Device(std::size_t n) {
        require(n != 0,"zero allocation");
        check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T)),"cudaMalloc");
    }
    ~Device() { if (ptr) cudaFree(ptr); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& data) {
        check(cudaMemcpy(ptr,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice),"upload");
    }
    std::vector<T> get(std::size_t n) const {
        std::vector<T> dst(n);
        check(cudaMemcpy(dst.data(),ptr,n*sizeof(T),cudaMemcpyDeviceToHost),"download");
        return dst;
    }
};
template<class T> T cast(float x) { return static_cast<T>(x); }
template<> __nv_bfloat16 cast<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template<class T> float asfloat(T x) { return static_cast<float>(x); }
template<> float asfloat<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template<class T> constexpr ink::ForwardScalar scalar_type();
template<> constexpr ink::ForwardScalar scalar_type<float>() { return ink::ForwardScalar::float32; }
template<> constexpr ink::ForwardScalar scalar_type<__nv_bfloat16>() { return ink::ForwardScalar::bfloat16; }

void compare(const std::vector<double>& exp, const std::vector<float>& got,
             const std::string& name, double atol=5e-5,double rtol=4e-5) {
    require(exp.size()==got.size(),"comparison lengths");
    double max_error=0.;
    for(std::size_t i=0;i<exp.size();++i) {
        const double v=got[i], r=exp[i];
        if (!std::isfinite(v) || std::abs(v-r) > atol + rtol*std::abs(r)) {
            throw std::runtime_error(name+" mismatch at "+std::to_string(i)+
                " actual="+std::to_string(v)+" expected="+std::to_string(r));
        }
        max_error=std::max(max_error,std::abs(v-r));
    }
    std::cout << "PASS " << name << " elements=" << got.size()
              << " max_error=" << max_error << '\n';
}
template<class T> std::vector<float> convert(const std::vector<T>& words) {
    std::vector<float> result(words.size());
    for (std::size_t i=0;i<words.size();++i)result[i]=asfloat(words[i]);
    return result;
}
// Independent, *full-prefill* scalar double implementation (no state reuse).
template<class T> std::vector<double> oracle(const std::vector<T>& x,
             const std::vector<float>& w,const std::vector<float>& dyn,
             const std::vector<std::uint8_t>& mask,const std::vector<std::int32_t>& ids,
             int batch,int seq,int channels,int heads,int taps,bool dynamic) {
    std::vector<double> out(x.size());
    const double scale=1/std::sqrt(static_cast<double>(dynamic?2*taps+1:taps+1));
    for (int b=0;b<batch;++b)for (int t=0;t<seq;++t)for (int c=0;c<channels;++c) {
        const std::size_t dst=(static_cast<std::size_t>(b)*seq+t)*channels+c;
        double sum=0;
        for(int lag=0;lag<taps;++lag) {
            const int src=t-lag;
            if(src<0)continue;
            const std::size_t bt=static_cast<std::size_t>(b)*seq+src;
            if(!mask.empty() && mask[bt]==0)continue;
            if(dynamic && !ids.empty() && ids[bt]!=ids[static_cast<std::size_t>(b)*seq+t])continue;
            double value=asfloat(x[bt*channels+c]);
            double weight=w[static_cast<std::size_t>(c)*taps+(dynamic?lag:taps-1-lag)];
            if(dynamic){const int head=c/(channels/heads);
                weight+=dyn[(static_cast<std::size_t>(b)*seq+t)*heads*taps+head*taps+lag];}
            sum+=value*weight;
        }
        out[dst]=asfloat(cast<T>(static_cast<float>((sum+asfloat(x[dst]))*scale)));
    }
    return out;
}

template<class T> void exercise(int batch,int sequence,int channels,int heads,
                                bool dynamic,bool use_mask,bool use_ids) {
    const int taps=4;
    const std::size_t total=static_cast<std::size_t>(batch)*sequence*channels;
    const std::size_t timesteps=static_cast<std::size_t>(batch)*sequence;
    std::vector<T> x(total);
    std::vector<float> w(static_cast<std::size_t>(channels)*taps);
    std::vector<float> dyn(dynamic?timesteps*heads*taps:1,0.f);
    std::vector<std::uint8_t> mask(timesteps,1);
    std::vector<std::int32_t> ids(timesteps,1);
    for (std::size_t i=0;i<x.size();++i)
        x[i]=cast<T>((static_cast<int>(i*17%79)-39)*0.017f);
    for (std::size_t i=0;i<w.size();++i)
        w[i]=(static_cast<int>(i*23%43)-21)*0.007f;
    for (std::size_t i=0;i<dyn.size();++i)
        dyn[i]=(static_cast<int>(i*11%37)-18)*0.003f;
    for (std::size_t i=0;i<mask.size();++i) {
        if(use_mask && (i%7==1 || i%7==5)) mask[i]=0;
        if(use_ids) ids[i]=static_cast<std::int32_t>(i%sequence/3);
    }
    const auto expected=oracle(x,w,dyn,use_mask?mask:std::vector<std::uint8_t>{},
        use_ids?ids:std::vector<std::int32_t>{},batch,sequence,channels,heads,taps,dynamic);
    Device<T> dx(total),dy(total),dchunk(total);
    Device<float> dw(w.size()),ddyn(dyn.size());
    Device<std::uint8_t> dm(mask.size());Device<std::int32_t> di(ids.size());
    dx.put(x);dw.put(w);ddyn.put(dyn);dm.put(mask);di.put(ids);
    ink::ConvCache48 cache;
    check(cache.create(batch,channels,taps),"cache create");
    const auto mode=dynamic?ink::ConvMode5C::dynamic_kv:ink::ConvMode5C::static_causal;
    const auto dtype=scalar_type<T>();
    const int conv=dynamic?0:2;
    const int layer=dynamic?11:6;
    check(cache.run(layer,0,conv,mode,dx.ptr,dw.ptr,dynamic?ddyn.ptr:nullptr,
          use_mask?dm.ptr:nullptr,use_ids?di.ptr:nullptr,
          dy.ptr,sequence,channels,dynamic?heads:0,dtype),"full launch");
    check(cudaDeviceSynchronize(),"full sync");
    const auto full=convert(dy.get(total));
    compare(expected,full,"native conv oracle"+(std::string(dynamic?" dynamic":" static"))+
            " B="+std::to_string(batch)+" C="+std::to_string(channels)+
            " dtype="+(std::is_same_v<T,float>?"FP32":"BF16"),
            std::is_same_v<T,float>?2e-5:0.016,
            std::is_same_v<T,float>?2e-5:0.012);

    // B>1 slices are not contiguous in [B,T,C], so chunk parity is checked
    // for B==1; multi-batch correctness is covered by the full oracle above.
    if(batch==1) {
        check(cache.reset(),"reset before chunks");
        const int pieces[]={2,1,3,1,2,3};
        int pos=0;
        for(int part:pieces) {
            if(pos>=sequence)break;
            const int len=std::min(part,sequence-pos);
            check(cache.run(layer,0,conv,mode,dx.ptr+static_cast<std::size_t>(pos)*channels,
                dw.ptr,dynamic?ddyn.ptr+static_cast<std::size_t>(pos)*heads*taps:nullptr,
                use_mask?dm.ptr+pos:nullptr,use_ids?di.ptr+pos:nullptr,
                dchunk.ptr+static_cast<std::size_t>(pos)*channels,
                len,channels,dynamic?heads:0,dtype),"chunk launch");
            pos+=len;
        }
        if(pos<sequence) {
            check(cache.run(layer,0,conv,mode,dx.ptr+static_cast<std::size_t>(pos)*channels,
                dw.ptr,dynamic?ddyn.ptr+static_cast<std::size_t>(pos)*heads*taps:nullptr,
                use_mask?dm.ptr+pos:nullptr,use_ids?di.ptr+pos:nullptr,
                dchunk.ptr+static_cast<std::size_t>(pos)*channels,
                sequence-pos,channels,dynamic?heads:0,dtype),"last chunk launch");
        }
        check(cudaDeviceSynchronize(),"chunk sync");
        const auto chunks=convert(dchunk.get(total));
        std::vector<double> full_d(full.begin(),full.end());
        compare(full_d,chunks,"prefill vs chunks C="+std::to_string(channels),
                std::is_same_v<T,float>?2e-6:0.0,0.0);
    }
}

void slot_and_validation() {
    std::vector<bool> seen(48,false);
    for(int layer=0;layer<24;++layer)for(int pass=0;pass<(layer>=6&&layer<18?3:1);++pass) {
        const int slot=ink::inkling2_cache_slot(layer,pass);
        require(slot>=0&&slot<48&&!seen[slot],"48-slot mapping collision");seen[slot]=true;
    }
    require(std::all_of(seen.begin(),seen.end(),[](bool v){return v;}),"48 slots incomplete");
    require(ink::inkling2_cache_slot(6,1)==24 && ink::inkling2_cache_slot(6,2)==36,
            "bad loop cache offset");
    require(ink::inkling2_cache_slot(23,1)==-1 && ink::inkling2_cache_slot(6,3)==-1,
            "bad slot accepted");
    ink::ConvCache48 cache;
    require(cache.create(0,4,4)==cudaErrorInvalidValue,"bad create accepted");
    check(cache.create(1,4,4),"cache create slot test");
    std::vector<float> one(4,1.f),zero(4,0.f),weights(16,0.f);
    for(int c=0;c<4;++c)weights[c*4+2]=1.f; // prior token at lag 1
    Device<float> done(4), dzero(4),dw(16), yempty(4),ypop(4);
    done.put(one);dzero.put(zero);dw.put(weights);
    require(cache.run(6,3,2,ink::ConvMode5C::static_causal,done.ptr,dw.ptr,
                      nullptr,nullptr,nullptr,ypop.ptr,1,4,0,
                      ink::ForwardScalar::float32)==cudaErrorInvalidValue,
            "invalid core pass accepted");
    require(cache.run(11,0,0,ink::ConvMode5C::static_causal,done.ptr,dw.ptr,
                      nullptr,nullptr,nullptr,ypop.ptr,1,4,0,
                      ink::ForwardScalar::float32)==cudaErrorInvalidValue,
            "full-attention static K/V unexpectedly accepted");
    require(cache.run(6,0,2,ink::ConvMode5C::static_causal,done.ptr,dw.ptr,
                      nullptr,nullptr,nullptr,done.ptr,1,4,0,
                      ink::ForwardScalar::float32)==cudaErrorInvalidValue,
            "in-place input/output accepted");
    check(cache.run(6,0,2,ink::ConvMode5C::static_causal,done.ptr,dw.ptr,
                    nullptr,nullptr,nullptr,ypop.ptr,1,4,0,
                    ink::ForwardScalar::float32),"fill slot 6");
    check(cache.run(6,1,2,ink::ConvMode5C::static_causal,dzero.ptr,dw.ptr,
                    nullptr,nullptr,nullptr,yempty.ptr,1,4,0,
                    ink::ForwardScalar::float32),"empty slot 24");
    check(cache.run(6,0,2,ink::ConvMode5C::static_causal,dzero.ptr,dw.ptr,
                    nullptr,nullptr,nullptr,ypop.ptr,1,4,0,
                    ink::ForwardScalar::float32),"revisit slot 6");
    check(cudaDeviceSynchronize(),"slot sync");
    require(std::abs(yempty.get(4)[0])<1e-7,"loop pass 1 leaked history from pass 0");
    require(ypop.get(4)[0]>0.2,"loop pass 0 lost its own history");
    check(cache.reset(),"reset cache");
    check(cache.run(6,0,2,ink::ConvMode5C::static_causal,dzero.ptr,dw.ptr,
                    nullptr,nullptr,nullptr,ypop.ptr,1,4,0,
                    ink::ForwardScalar::float32),"after reset");
    check(cudaDeviceSynchronize(),"reset sync");
    require(std::abs(ypop.get(4)[0])<1e-7,"reset didn't clear state");
    std::cout<<"PASS 48-slot mapping, per-pass isolation, reset, invalid argument contracts\n";
}
} // namespace
int main() {
    try {
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p,0),"GPU properties");
        std::cout << "GPU " << p.name << " sm_" << p.major << p.minor << '\n';
        slot_and_validation();
        exercise<float>(1,12,19,1,false,true,false);
        exercise<float>(2,7,128,1,false,true,false);
        exercise<float>(1,12,1024,8,true,true,true);
        exercise<float>(2,7,1024,8,true,true,true);
        exercise<__nv_bfloat16>(1,12,1024,8,true,true,true);
        exercise<float>(1,9,2304,1,false,true,false);
        exercise<__nv_bfloat16>(1,9,2304,1,false,true,false);
        std::cout << "OK Stage 5C: native CUDA dynamic K/V conv + static streams + 48-slot state parity\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL Stage 5C: " << e.what() << '\n';
        return 1;
    }
}

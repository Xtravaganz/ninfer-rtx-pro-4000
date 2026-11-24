// Stage 5B native CUDA regression executable. Numerics checked independently
// against scalar CPU double references; original Inkling head/channel geometry.
#include "models/inkling2/forward_primitives.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ink = ninfer::models::inkling2;
namespace {
void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
void cuda_check(cudaError_t result, const char* where) {
    if (result != cudaSuccess) throw std::runtime_error(
        std::string(where) + ": " + cudaGetErrorString(result));
}
template<typename T> struct Device {
    T* data = nullptr;
    explicit Device(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)),"malloc");
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
    void upload(const std::vector<T>& src) {
        cuda_check(cudaMemcpy(data,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice),"upload");
    }
    void download(std::vector<T>& dst) {
        cuda_check(cudaMemcpy(dst.data(),data,dst.size()*sizeof(T),cudaMemcpyDeviceToHost),"download");
    }
};

template<typename T> float flt(T v) { return static_cast<float>(v); }
template<> float flt<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template<typename T> T quant(float x) { return static_cast<T>(x); }
template<> __nv_bfloat16 quant<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <typename T> constexpr ink::ForwardScalar kind();
template <> constexpr ink::ForwardScalar kind<float>() { return ink::ForwardScalar::float32; }
template <> constexpr ink::ForwardScalar kind<__nv_bfloat16>() { return ink::ForwardScalar::bfloat16; }
template<typename T> void close(const std::vector<T>& actual,
                                 const std::vector<double>& expected,
                                 double abs_tol, double rel_tol, const char* label) {
    require(actual.size()==expected.size(),"comparison length mismatch");
    double max_error = 0.;
    for (std::size_t i=0;i<actual.size();++i) {
        const double a=static_cast<double>(flt(actual[i]));
        const double e=expected[i];
        if (!std::isfinite(a) || std::abs(a-e)>abs_tol+rel_tol*std::abs(e)) {
            throw std::runtime_error(std::string(label)+" mismatch index="+std::to_string(i)+
                " got="+std::to_string(a)+" expected="+std::to_string(e));
        }
        max_error=std::max(max_error,std::abs(a-e));
    }
    std::cout << "PASS " << label << " elements=" << actual.size()
              << " max_error=" << max_error << '\n';
}

template<typename T> void test_rms(int rows, int hidden, bool weighted) {
    constexpr float eps=1e-6f;
    std::vector<T> x(static_cast<std::size_t>(rows)*hidden), result(x.size());
    std::vector<float> weight(hidden);
    for (int j=0;j<hidden;++j) weight[j]=0.6f+0.025f*(j%23);
    for (std::size_t i=0;i<x.size();++i)
        x[i]=quant<T>(static_cast<float>((static_cast<int>(i*17%79)-39))*0.041f);
    std::vector<double> expected(x.size());
    for(int r=0;r<rows;++r) {
        double sq=0;
        for (int j=0;j<hidden;++j) {
            const double v=flt(x[static_cast<std::size_t>(r)*hidden+j]); sq+=v*v;
        }
        const double factor=1/std::sqrt(sq/hidden+eps);
        for (int j=0;j<hidden;++j) {
            double norm=flt(x[static_cast<std::size_t>(r)*hidden+j])*factor;
            // Stage-5A unweighted RMSNorm casts before learned weight.
            if (weighted) norm=flt(quant<T>(static_cast<float>(norm)))*weight[j];
            expected[static_cast<std::size_t>(r)*hidden+j]=flt(quant<T>(static_cast<float>(norm)));
        }
    }
    Device<T> dx(x.size()), dy(x.size()); Device<float> dw(weight.size());
    dx.upload(x); dw.upload(weight);
    cuda_check(ink::launch_rmsnorm_forward(dx.data,weighted ? dw.data : nullptr,dy.data,
        rows,hidden,eps,weighted,kind<T>()),"rms launch");
    cuda_check(cudaDeviceSynchronize(),"rms sync"); dy.download(result);
    close(result,expected, std::is_same_v<T,float>?0.0002:0.011,
        std::is_same_v<T,float>?0.0001:0.011, "RMSNorm");
}

template<typename T> void test_conv(int batch,int seq,int channels,int taps,bool use_mask) {
    const std::size_t total=static_cast<std::size_t>(batch)*seq*channels;
    std::vector<T> x(total), result(total);
    std::vector<float> weight(static_cast<std::size_t>(channels)*taps);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(batch)*seq,1);
    for (std::size_t i=0;i<total;++i)
        x[i]=quant<T>(static_cast<float>(static_cast<int>(i*11%91)-45)*0.032f);
    for(std::size_t i=0;i<weight.size();++i)
        weight[i]=static_cast<float>(static_cast<int>(i*13%19)-9)*0.025f;
    if(use_mask) for (std::size_t i=0;i<mask.size();++i)
        if(i%5==2)mask[i]=0;
    std::vector<double> expected(total);
    for(int b=0;b<batch;++b)for(int t=0;t<seq;++t)for(int c=0;c<channels;++c){
        double sum=0;
        const std::size_t dst=(static_cast<std::size_t>(b)*seq+t)*channels+c;
        for(int j=0;j<taps;++j){
            const int prev=t+j-(taps-1);
            if(prev<0 || (use_mask && mask[static_cast<std::size_t>(b)*seq+prev]==0))continue;
            sum+=flt(x[(static_cast<std::size_t>(b)*seq+prev)*channels+c])*
                 weight[static_cast<std::size_t>(c)*taps+j];
        }
        const double output=(sum+flt(x[dst]))/std::sqrt(static_cast<double>(taps+1));
        expected[dst]=flt(quant<T>(static_cast<float>(output)));
    }
    Device<T> dx(total),dy(total);Device<float> dw(weight.size());
    Device<std::uint8_t> dm(mask.size());
    dx.upload(x);dw.upload(weight);dm.upload(mask);
    cuda_check(ink::launch_static_conv_forward(dx.data,dw.data,
        use_mask ? dm.data : nullptr,dy.data,batch,seq,channels,taps,kind<T>()),
        "static conv launch");
    cuda_check(cudaDeviceSynchronize(),"static conv sync");dy.download(result);
    close(result,expected, std::is_same_v<T,float>?0.00005:0.009,
        std::is_same_v<T,float>?0.00005:0.009,"static causal convolution");
}

template<typename T> void test_attention(int batch, int hq, int hkv, int keys,
                                         int dim, bool all_masked) {
    const std::size_t qn=static_cast<std::size_t>(batch)*hq*dim;
    const std::size_t kn=static_cast<std::size_t>(batch)*hkv*keys*dim;
    const std::size_t logits=static_cast<std::size_t>(batch)*hq*keys;
    std::vector<T> q(qn),k(kn),v(kn),out(qn);
    std::vector<float> bias(logits),sinks(static_cast<std::size_t>(batch)*hq);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(batch)*keys);
    for(std::size_t i=0;i<qn;++i) q[i]=quant<T>(static_cast<float>(static_cast<int>(i*17%61)-30)*0.016f);
    for(std::size_t i=0;i<kn;++i) {
        k[i]=quant<T>(static_cast<float>(static_cast<int>(i*19%47)-23)*0.017f);
        v[i]=quant<T>(static_cast<float>(static_cast<int>(i*23%59)-29)*0.021f);
    }
    for(std::size_t i=0;i<logits;++i)bias[i]=static_cast<float>(static_cast<int>(i*7%31)-15)*0.014f;
    for(std::size_t i=0;i<sinks.size();++i) sinks[i]=static_cast<float>(i%9)*0.06f-0.1f;
    for (std::size_t i=0;i<mask.size();++i) mask[i]=(!all_masked && i%4!=2)?1:0;
    std::vector<double> expected(qn);
    for(int b=0;b<batch;++b)for(int h=0;h<hq;++h){
        const int kh=h/(hq/hkv);
        const std::size_t qbase=(static_cast<std::size_t>(b)*hq+h)*dim;
        const std::size_t kvbase=(static_cast<std::size_t>(b)*hkv+kh)*keys*dim;
        const std::size_t biasbase=(static_cast<std::size_t>(b)*hq+h)*keys;
        const double sink=sinks[static_cast<std::size_t>(b)*hq+h];
        std::vector<double> logit(keys,-std::numeric_limits<double>::infinity());
        double maxv=sink;
        for(int token=0;token<keys;++token){
            if(!mask[static_cast<std::size_t>(b)*keys+token]) continue;
            double dot=0;
            for(int d=0;d<dim;++d)
                dot+=flt(q[qbase+d])*flt(k[kvbase+static_cast<std::size_t>(token)*dim+d]);
            logit[token]=dot/std::sqrt(static_cast<double>(dim))+bias[biasbase+token];
            maxv=std::max(maxv,logit[token]);
        }
        double denom=std::exp(sink-maxv);
        for(int token=0;token<keys;++token)if(std::isfinite(logit[token]))denom+=std::exp(logit[token]-maxv);
        for(int d=0;d<dim;++d){
            double num=0;
            for(int token=0;token<keys;++token)if(std::isfinite(logit[token]))
                num+=std::exp(logit[token]-maxv)*flt(v[kvbase+static_cast<std::size_t>(token)*dim+d]);
            expected[qbase+d]=flt(quant<T>(static_cast<float>(num/denom)));
        }
    }
    Device<T> dq(qn),dk(kn),dv(kn),dy(qn);
    Device<float> dbias(logits),dsink(sinks.size());
    Device<std::uint8_t> dmask(mask.size());
    dq.upload(q);dk.upload(k);dv.upload(v);dbias.upload(bias);dsink.upload(sinks);dmask.upload(mask);
    cuda_check(ink::launch_sink_gqa_decode(dq.data,dk.data,dv.data,dbias.data,
       dsink.data,dmask.data,dy.data,batch,hq,hkv,keys,dim,kind<T>()),"gqa launch");
    cuda_check(cudaDeviceSynchronize(),"gqa sync");dy.download(out);
    close(out,expected, std::is_same_v<T,float>?0.0001:0.004,
        std::is_same_v<T,float>?0.0001:0.007,"sink GQA decode");
}

void invalid_args() {
    require(ink::launch_rmsnorm_forward(nullptr,nullptr,nullptr,1,2304,1e-6f,false,
        ink::ForwardScalar::float32)==cudaErrorInvalidValue,"RMS null not rejected");
    require(ink::launch_static_conv_forward(nullptr,nullptr,nullptr,nullptr,1,1,2304,4,
        ink::ForwardScalar::float32)==cudaErrorInvalidValue,"conv null not rejected");
    require(ink::launch_sink_gqa_decode(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        nullptr,1,24,7,10,128,ink::ForwardScalar::float32)==cudaErrorInvalidValue,
        "GQA invalid head group not rejected");
    Device<float> dx(8),dy(8);
    require(ink::launch_rmsnorm_forward(dx.data,nullptr,dy.data,1,8,0.f,false,
        ink::ForwardScalar::float32)==cudaErrorInvalidValue,"RMS eps not rejected");
    require(ink::launch_rmsnorm_forward(dx.data,nullptr,dy.data,1,8,1e-6f,true,
        ink::ForwardScalar::float32)==cudaErrorInvalidValue,"weighted norm without weights accepted");
    require(ink::launch_static_conv_forward(dx.data,dx.data,nullptr,dy.data,1,1,8,0,
        ink::ForwardScalar::float32)==cudaErrorInvalidValue,"conv bad taps accepted");
    require(ink::launch_sink_gqa_decode(dx.data,dx.data,dx.data,nullptr,dx.data,nullptr,
        dy.data,1,24,8,1,512,ink::ForwardScalar::float32)==cudaErrorInvalidValue,
        "attention head dimension greater than 256 accepted");
    std::cout << "PASS input contract / invalid arguments\n";
}
} // namespace

int main() {
    try {
        int count=0;
        cuda_check(cudaGetDeviceCount(&count),"device count");
        require(count>0,"Stage 5B requires CUDA device");
        cudaDeviceProp props{};
        cuda_check(cudaGetDeviceProperties(&props,0),"properties");
        std::cout << "GPU: " << props.name << " sm_" << props.major << props.minor << '\n';
        invalid_args();
        test_rms<float>(3,17,false);
        test_rms<float>(2,2304,true);
        test_rms<__nv_bfloat16>(3,2304,true);
        test_rms<__nv_bfloat16>(1,128,false);
        test_conv<float>(2,9,19,4,true);
        test_conv<float>(1,7,2304,4,false);
        test_conv<__nv_bfloat16>(2,6,128,4,true);
        test_attention<float>(1,6,2,7,16,false);
        test_attention<float>(1,6,2,5,16,true);
        test_attention<float>(2,24,8,33,128,false);
        test_attention<__nv_bfloat16>(1,24,8,9,128,false);
        std::cout << "OK Stage 5B: native CUDA RMSNorm / short convolution / sink GQA decode\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL Stage 5B: " << e.what() << '\n';
        return 1;
    }
}

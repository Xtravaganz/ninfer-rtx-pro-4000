// Stage 6K/L: native reusable CUDA graphs for 6H MoE, and native causal
// short-convolution state for all 48 logical Inkling slots. GPU 0 only.
// This deliberately does NOT claim to implement the complete 48-visit model.
#include "models/inkling2/native_resident_stage6gh.h"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>

namespace {
constexpr int Layers = 24, Slots = 48, Paths = 4, Hidden = 2304, Taps = 4;
// CUDA stream handle 0 is the valid legacy/default stream.
constexpr int RouterRows = 128, RouterElements = RouterRows * Hidden;

// A captured graph is pinned to the exact store, workspace and buffers used
// during capture. A single stream and one pending execution per instance only.
struct GraphSlot {
    float *input=nullptr, *output=nullptr, *router=nullptr, *bias=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    bool ready=false;
    void clear() noexcept {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (input) cudaFree(input);
        if (output) cudaFree(output);
        if (router) cudaFree(router);
        if (bias) cudaFree(bias);
        executable=nullptr;graph=nullptr;input=output=router=bias=nullptr;ready=false;
    }
};
struct GraphRuntime {
    void* store;
    void* workspace;
    int device=-1;
    cudaStream_t capture_stream=nullptr;
    std::array<GraphSlot,Layers> slot{};
    int captures=0;
    std::uint64_t replays=0;
    int last_capture_phase=0, last_capture_error=0;
    std::mutex mutex;
    GraphRuntime(void* s, void* w):store(s),workspace(w) {
        if (cudaGetDevice(&device)==cudaSuccess && device==0)
            cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking);
    }
    ~GraphRuntime() noexcept {
        if(device>=0) {
            cudaSetDevice(device);
            // Shutdown only: graphs may have replayed on external Torch streams.
            cudaDeviceSynchronize();
        }
        for(auto& s:slot) s.clear();
        if(capture_stream)cudaStreamDestroy(capture_stream);
    }
    int fail(int phase,int error) noexcept {
        last_capture_phase=phase;
        last_capture_error=error;
        return error;
    }
};
int allocate(GraphSlot& s) noexcept {
    cudaError_t e;
    if ((e=cudaMalloc(reinterpret_cast<void**>(&s.input),Hidden*sizeof(float)))!=cudaSuccess)return int(e);
    if ((e=cudaMalloc(reinterpret_cast<void**>(&s.output),Hidden*sizeof(float)))!=cudaSuccess)return int(e);
    if ((e=cudaMalloc(reinterpret_cast<void**>(&s.router),RouterElements*sizeof(float)))!=cudaSuccess)return int(e);
    if ((e=cudaMalloc(reinterpret_cast<void**>(&s.bias),RouterRows*sizeof(float)))!=cudaSuccess)return int(e);
    return 0;
}
int capture(GraphRuntime* runtime,int layer,std::uint64_t router,std::uint64_t bias,
            float scale,cudaStream_t caller_stream) noexcept {
    if (!runtime || !router || !bias || layer<0 || layer>=Layers ||
        !std::isfinite(scale) || scale<=0) return 1001;
    std::lock_guard<std::mutex> lock(runtime->mutex);
    runtime->last_capture_phase=0;
    runtime->last_capture_error=0;
    if (runtime->slot[layer].ready) return runtime->fail(0,1002);
    int device=-1;
    if (cudaGetDevice(&device)!=cudaSuccess || device!=runtime->device)
        return runtime->fail(0,1003);
    if (!runtime->capture_stream) return runtime->fail(0,1005);
    GraphSlot& s=runtime->slot[layer];
    // Source router/bias may have been produced on the PyTorch current stream.
    // This is a one-time initialization wait, never in warm decode.
    cudaError_t e=cudaStreamSynchronize(caller_stream);
    if (e!=cudaSuccess) return runtime->fail(1,int(e));
    if (const auto rc=allocate(s)) {s.clear();return runtime->fail(2,rc);}
    const auto capture_stream=runtime->capture_stream;
    e=cudaMemcpyAsync(s.router,reinterpret_cast<const void*>(router),
                      RouterElements*sizeof(float),cudaMemcpyDeviceToDevice,capture_stream);
    if (e==cudaSuccess)
        e=cudaMemcpyAsync(s.bias,reinterpret_cast<const void*>(bias),
                          RouterRows*sizeof(float),cudaMemcpyDeviceToDevice,capture_stream);
    if (e!=cudaSuccess) {s.clear();return runtime->fail(3,int(e));}
    e=cudaStreamSynchronize(capture_stream);
    if (e!=cudaSuccess) {s.clear();return runtime->fail(4,int(e));}
    e=cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal);
    if (e!=cudaSuccess) {s.clear();return runtime->fail(5,int(e));}
    const int rc=inkling6h_run(runtime->store,runtime->workspace,layer,
        reinterpret_cast<std::uint64_t>(s.input),reinterpret_cast<std::uint64_t>(s.router),
        reinterpret_cast<std::uint64_t>(s.bias),scale,
        reinterpret_cast<std::uint64_t>(s.output),reinterpret_cast<std::uint64_t>(capture_stream));
    // End capture even if the native call reports an error, to release CUDA's
    // capture state before freeing any of the graph-owned buffers.
    e=cudaStreamEndCapture(capture_stream,&s.graph);
    if (rc || e!=cudaSuccess || !s.graph) {
        s.clear();
        if(rc)return runtime->fail(6,rc);
        return runtime->fail(7,int(e==cudaSuccess?cudaErrorInvalidValue:e));
    }
    e=cudaGraphInstantiate(&s.executable,s.graph,0);
    if (e!=cudaSuccess) {s.clear();return runtime->fail(8,int(e));}
    s.ready=true;
    ++runtime->captures;
    return 0;
}
int replay(GraphRuntime* runtime,int layer,std::uint64_t x,std::uint64_t out,
           cudaStream_t stream) noexcept {
    if (!runtime || layer<0 || layer>=Layers || !x || !out)return 1001;
    std::lock_guard<std::mutex> lock(runtime->mutex);
    if (!runtime->slot[layer].ready)return 1004;
    const GraphSlot& s=runtime->slot[layer];
    cudaError_t e=cudaMemcpyAsync(s.input,reinterpret_cast<const void*>(x),
                                  Hidden*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    if(e!=cudaSuccess)return int(e);
    e=cudaGraphLaunch(s.executable,stream);
    if(e!=cudaSuccess)return int(e);
    e=cudaMemcpyAsync(reinterpret_cast<void*>(out),s.output,Hidden*sizeof(float),
                      cudaMemcpyDeviceToDevice,stream);
    if(e!=cudaSuccess)return int(e);
    ++runtime->replays;
    return 0;
}

struct ConvRuntime {
    float* history=nullptr;
    int device=-1;
    cudaStream_t stream=nullptr;
    ConvRuntime(){cudaGetDevice(&device);}
    ~ConvRuntime() noexcept {
        cudaSetDevice(device);
        if(stream) cudaStreamSynchronize(stream);
        if(history)cudaFree(history);
    }
};
// history: oldest -> newest, state for every logical (slot,path).
// Each thread owns one channel; a batch can be processed in one launch,
// preserving causal history within each thread.
__global__ void causal_conv(float* history, const float* input, const float* taps,
                            float* output,int channels,int tokens,int state,int dynamic) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=channels)return;
    float* h=history+(std::size_t(state)*3*Hidden+c);
    float h0=h[0],h1=h[Hidden],h2=h[2*Hidden];
    for(int t=0;t<tokens;++t){
        const float x=input[std::size_t(t)*channels+c];
        const float* w=taps+std::size_t(dynamic ? t*channels+c : c)*Taps;
        float value;
        if (dynamic) {
            // Tap order is [current, t-1, t-2, t-3].
            value=x*w[0]+h2*w[1]+h1*w[2]+h0*w[3];
            output[std::size_t(t)*channels+c]=(value+x)*(1.f/3.f); // sqrt(2*4+1)
        } else {
            value=h0*w[0]+h1*w[1]+h2*w[2]+x*w[3];
            output[std::size_t(t)*channels+c]=(value+x)*0.447213595499957939f;
        }
        h0=h1;h1=h2;h2=x;
    }
    h[0]=h0;h[Hidden]=h1;h[2*Hidden]=h2;
}
} // namespace

extern "C" {
void* inkling6k_graph_create(void* store,void* workspace) noexcept {
    if (!store || !workspace)return nullptr;
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess || device!=0)return nullptr;
    auto* graph=new(std::nothrow) GraphRuntime(store,workspace);
    if(graph && !graph->capture_stream){delete graph;return nullptr;}
    return graph;
}
void inkling6k_graph_destroy(void* value) noexcept {delete static_cast<GraphRuntime*>(value);}
int inkling6k_capture(void* p,int layer,std::uint64_t router,std::uint64_t bias,
                      float scale,std::uint64_t stream) noexcept {
    return capture(static_cast<GraphRuntime*>(p),layer,router,bias,scale,
                   reinterpret_cast<cudaStream_t>(stream));
}
int inkling6k_replay(void* p,int layer,std::uint64_t input,std::uint64_t output,
                     std::uint64_t stream) noexcept {
    return replay(static_cast<GraphRuntime*>(p),layer,input,output,
                  reinterpret_cast<cudaStream_t>(stream));
}
int inkling6k_captured_layers(void* p) noexcept {
    return p?static_cast<GraphRuntime*>(p)->captures:0;
}
std::uint64_t inkling6k_replay_count(void* p) noexcept {
    return p?static_cast<GraphRuntime*>(p)->replays:0;
}
// Diagnostic phases: 1=source wait, 2=allocation, 3=weight copy,
// 4=private-stream wait, 5=BeginCapture, 6=native MoE,
// 7=EndCapture, 8=Instantiate.  Zero means no capture failure.
int inkling6k_last_capture_phase(void* p) noexcept {
    return p?static_cast<GraphRuntime*>(p)->last_capture_phase:0;
}
int inkling6k_last_capture_error(void* p) noexcept {
    return p?static_cast<GraphRuntime*>(p)->last_capture_error:0;
}
void* inkling6l_conv_create() noexcept {
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess || device!=0)return nullptr;
    auto* runtime=new(std::nothrow) ConvRuntime();
    if(!runtime)return nullptr;
    const std::size_t bytes=std::size_t(Slots)*Paths*(Taps-1)*Hidden*sizeof(float);
    auto e=cudaMalloc(reinterpret_cast<void**>(&runtime->history),bytes);
    if(e==cudaSuccess)e=cudaMemset(runtime->history,0,bytes);
    if(e!=cudaSuccess){delete runtime;return nullptr;}
    return runtime;
}
void inkling6l_conv_destroy(void* p) noexcept {delete static_cast<ConvRuntime*>(p);}
int inkling6l_conv_apply(void* p,int slot,int path,std::uint64_t input,
                        std::uint64_t taps,std::uint64_t output,int channels,
                        int tokens,int dynamic,std::uint64_t stream) noexcept {
    auto* c=static_cast<ConvRuntime*>(p);
    if(!c||slot<0||slot>=Slots||path<0||path>=Paths||channels<1||channels>Hidden||
       tokens<1||tokens>4096||(dynamic!=0&&dynamic!=1)||!input||!taps||!output)
       return int(cudaErrorInvalidValue);
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess || device!=c->device)return 1003;
    const auto s=reinterpret_cast<cudaStream_t>(stream);
    if(c->stream && c->stream!=s)return 1004;
    c->stream=s;
    causal_conv<<<(channels+255)/256,256,0,s>>>(c->history,
        reinterpret_cast<const float*>(input),reinterpret_cast<const float*>(taps),
        reinterpret_cast<float*>(output),channels,tokens,slot*Paths+path,dynamic);
    return int(cudaGetLastError());
}
}

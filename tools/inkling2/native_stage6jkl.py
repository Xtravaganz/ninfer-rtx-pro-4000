"""Native Inkling 6J/K/L bridge (opt-in, GPU0 only).

J: preload ALL 24 native compressed expert banks before the first forward.
K: optional native CUDA Graph replay for the existing 6H GPU-routed MoE;
   graph owns stable activation/router/output buffers and shares the native
   expert store. Capture happens once before the first token.
L: native CUDA persistent causal static/dynamic short convolution for 48 slots;
   also provides a multiple-token operator, without claiming whole-model prefill.

Not a complete native Inkling executor: Torch retains attention, norm, residual
controllers, cross-pass messages, final logits, and sequential token scheduling.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from time import perf_counter

from native_resident_stage6ghi import ResidentCarrierWeights
from streaming_stage5q import IncrementalCache48

U64=ctypes.c_uint64
PTR=ctypes.c_void_p
INT=ctypes.c_int
FLOAT=ctypes.c_float


def configure_api(library):
    """Register exact signatures once, no implicit ctypes pointer conversions."""
    def fn(name,args,result):
        f=getattr(library,name);f.argtypes=args;f.restype=result
    fn('inkling6k_graph_create',[PTR,PTR],PTR)
    fn('inkling6k_graph_destroy',[PTR],None)
    fn('inkling6k_capture',[PTR,INT,U64,U64,FLOAT,U64],INT)
    fn('inkling6k_replay',[PTR,INT,U64,U64,U64],INT)
    fn('inkling6k_captured_layers',[PTR],INT)
    fn('inkling6k_replay_count',[PTR],U64)
    fn('inkling6l_conv_create',[],PTR)
    fn('inkling6l_conv_destroy',[PTR],None)
    fn('inkling6l_conv_apply',[PTR,INT,INT,U64,U64,U64,INT,INT,INT,U64],INT)
    return library


def require_device0(device):
    import torch
    if torch.device(device) != torch.device('cuda:0'):
        raise ValueError(f'Stage6J/K/L is pinned to cuda:0, received {device}')
    if not torch.cuda.is_available():
        raise RuntimeError('Stage6J/K/L requires NVIDIA CUDA device 0')
    torch.cuda.set_device(0)


class GraphMoE:
    def __init__(self,native):
        self.lib=configure_api(native.lib)
        self.handle=self.lib.inkling6k_graph_create(native.store,native.single)
        if not self.handle: raise RuntimeError('native Stage6K graph allocation failed (GPU0)')

    def capture(self,layer,router,bias,scale,stream):
        if not router.is_contiguous() or not bias.is_contiguous():
            raise ValueError('Stage6K graph capture needs contiguous resident router/bias')
        rc=self.lib.inkling6k_capture(self.handle,int(layer),router.data_ptr(),
                      bias.data_ptr(),float(scale),int(stream.cuda_stream))
        if rc:
            # Optional ABI: older 6K libraries can still be inspected safely.
            detail=''
            phase_fn=getattr(self.lib,'inkling6k_last_capture_phase',None)
            if phase_fn is not None:
                phase_fn.argtypes=[PTR]
                phase_fn.restype=INT
                phase=int(phase_fn(self.handle))
                names={1:'source_stream_wait',2:'allocate',3:'copy_router_bias',
                       4:'private_stream_wait',5:'begin_capture',
                       6:'native_moe',7:'end_capture',8:'graph_instantiate'}
                detail=f', phase={names.get(phase,phase)}'
            raise RuntimeError(f'native Stage6K graph capture failed layer={layer}, code={rc}{detail}')




    def run(self,x,out,layer,stream):
        if not x.is_contiguous() or not out.is_contiguous():
            raise ValueError('native CUDA graph requires contiguous input and output')
        rc=self.lib.inkling6k_replay(self.handle,int(layer),x.data_ptr(),out.data_ptr(),
                                     int(stream.cuda_stream))
        if rc: raise RuntimeError(f'native Stage6K graph replay failed layer={layer}, code={rc}')

    def metrics(self):
        return {'native_graph_layers':int(self.lib.inkling6k_captured_layers(self.handle)),
                'native_graph_replays':int(self.lib.inkling6k_replay_count(self.handle))}

    def close(self):
        if getattr(self,'handle',None):
            self.lib.inkling6k_graph_destroy(self.handle)
            self.handle=None


class PreloadedResidentWeights(ResidentCarrierWeights):
    """Keep resident GPU banks, separate explicit startup from forward metrics."""
    def __init__(self,reader,device,geometry,*,library_path,nonexpert_mib=4096,
                 verify_first=False,use_graph=False):
        import torch
        require_device0(device)
        self.graph=None
        self._preload_seconds=0.
        self._capture_seconds=0.
        self._graph_enabled=bool(use_graph)
        super().__init__(reader,torch.device('cuda:0'),geometry,
                         library_path=library_path,nonexpert_mib=nonexpert_mib,
                         verify_first=verify_first)
        try:
            t=perf_counter()
            for layer in range(24):
                self.native.upload(layer,reader)
            self._preload_seconds=perf_counter()-t
            assert self.native.lib.inkling6g_resident_layers(self.native.store)==24
            print(f'STAGE6J NATIVE PRELOAD: layers=24 seconds={self._preload_seconds:.3f} '
                  f'mib={self.native.metrics()["native_resident_mib"]:.2f} device=cuda:0',flush=True)
            if use_graph:
                t=perf_counter()
                self.graph=GraphMoE(self.native)
                stream=torch.cuda.current_stream(torch.device('cuda:0'))
                for layer in range(24):
                    p=f'model.llm.layers.{layer}.mlp.'
                    router=self.tensor(p+'gate.weight',(128,2304)).contiguous()
                    bias=self.tensor(p+'gate.bias',(128,)).contiguous()
                    self.graph.capture(layer,router,bias,geometry.route_scale,stream)
                self._capture_seconds=perf_counter()-t
                print(f'STAGE6K NATIVE GRAPH CAPTURE: layers=24 '
                      f'seconds={self._capture_seconds:.3f} device=cuda:0',flush=True)
        except BaseException:
            # No partially ready program may escape to the inference loop.
            self.release()
            raise

    def moe(self,hidden,layer):
        # Parent MoE method is already native and performs its own optional
        # first-MoE verification against the FP32 student reference. Redirect
        # only the *native execution* function, never the validation path.
        if self.graph is None:
            return super().moe(hidden,layer)
        original=self.native.run
        def graph_run(x,router,bias,route_scale,physical,out,stream):
            self.graph.run(x,out,physical,stream)
            self.native.calls_single+=1
        self.native.run=graph_run
        try: return super().moe(hidden,layer)
        finally: self.native.run=original

    def metrics(self):
        m=super().metrics()
        m['native_preload_s']=round(self._preload_seconds,3)
        m['native_graph_capture_s']=round(self._capture_seconds,3)
        m['native_graph_enabled']=self._graph_enabled
        if self.graph is not None:m.update(self.graph.metrics())
        return m

    def release(self):
        # Captured graph pointers must be released BEFORE native workspaces and
        # resident banks, which the graph execution refers to.
        if getattr(self,'graph',None) is not None:
            self.graph.close();self.graph=None
        if getattr(self,'native',None) is not None:
            super().release()


class ConvState:
    """Native CUDA convolution state, strictly bound to cuda:0, one stream."""
    def __init__(self,libpath):
        require_device0('cuda:0')
        self.lib=configure_api(ctypes.CDLL(str(Path(libpath).resolve())))
        self.handle=self.lib.inkling6l_conv_create()
        if not self.handle:raise RuntimeError('native Stage6L convolution state allocation failed')
        self.calls_single=0
        self.calls_batch=0

    def apply(self,slot,path,x,taps,out,*,dynamic=False):
        import torch
        if (x.device!=torch.device('cuda:0') or out.device!=x.device or taps.device!=x.device or
            x.dtype!=torch.float32 or taps.dtype!=torch.float32 or out.dtype!=torch.float32 or
            not (x.is_contiguous() and taps.is_contiguous() and out.is_contiguous()) or
            x.ndim!=2 or x.shape!=out.shape or x.shape[1]>2304 or
            taps.shape != ((x.shape[0],x.shape[1],4) if dynamic else (x.shape[1],4))):
            raise ValueError('native Stage6L convolution tensor/geometry/device mismatch')
        s=torch.cuda.current_stream(torch.device('cuda:0'))
        rc=self.lib.inkling6l_conv_apply(self.handle,int(slot),int(path),
            x.data_ptr(),taps.data_ptr(),out.data_ptr(),int(x.shape[1]),int(x.shape[0]),
            int(bool(dynamic)),int(s.cuda_stream))
        if rc:raise RuntimeError(f'Stage6L convolution CUDA error={rc}, slot={slot}, path={path}')
        if x.shape[0]==1:self.calls_single+=1
        else:self.calls_batch+=1

    def close(self):
        if getattr(self,'handle',None):
            self.lib.inkling6l_conv_destroy(self.handle);self.handle=None


def make_native_conv_cache(library_path):
    """Return an opt-in subclass of Stage5Q cache; no edits to existing runtime."""
    class NativeConvCache(IncrementalCache48):
        def __init__(self,kernel=4):
            if kernel!=4:raise ValueError('Stage6L only supports Inkling K=4')
            super().__init__(kernel=kernel)
            self.native_conv=ConvState(library_path)

        def _check(self,slot,path,x,channels):
            if not 0<=slot<48 or not 0<=path<4 or (slot,path) in self._conv_visited:
                raise ValueError('duplicate or invalid native convolution slot/path')
            self._check_slot(slot)
            if x.ndim!=3 or x.shape[:2]!=(1,1) or x.shape[-1]!=channels:
                raise ValueError('Stage6L requires [1,1,C] input')

        def static_conv(self,slot,path,x,taps):
            import torch
            c=x.shape[-1]
            self._check(slot,path,x,c)
            if taps.shape!=(c,4):raise ValueError('static convolution tap shape mismatch')
            out=torch.empty_like(x,dtype=torch.float32)
            self.native_conv.apply(slot,path,x.reshape(1,c).contiguous().float(),
                         taps.float().contiguous(),out.reshape(1,c),dynamic=False)
            self._conv_visited.add((slot,path))
            return out.to(x.dtype)

        def dynamic_conv(self,slot,path,x,dynamic_input,taps,projection,*,heads):
            import torch
            import torch.nn.functional as F
            c=x.shape[-1]
            self._check(slot,path,x,c)
            if (heads<1 or c%heads or dynamic_input.ndim!=3 or
                dynamic_input.shape[:2]!=(1,1) or taps.shape!=(4,c) or
                projection.shape!=(heads*4,dynamic_input.shape[-1])):
                raise ValueError('dynamic convolution geometry mismatch')
            # Preserve the exact dynamic projection and tap order of Stage5Q.
            delta=F.linear(dynamic_input.float(),projection.float()).reshape(heads,1,4)
            combined=(taps.float().T.reshape(heads,c//heads,4)+delta).reshape(1,c,4).contiguous()
            out=torch.empty_like(x,dtype=torch.float32)
            self.native_conv.apply(slot,path,x.reshape(1,c).contiguous().float(),
                         combined,out.reshape(1,c),dynamic=True)
            self._conv_visited.add((slot,path))
            return out.to(x.dtype)

        def close(self):
            if getattr(self,'native_conv',None) is not None:
                self.native_conv.close();self.native_conv=None

        def __del__(self):
            self.close()

    return NativeConvCache

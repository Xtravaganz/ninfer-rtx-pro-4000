"""Stage 5W: opt-in native CUDA MoE for Stage-5U 48-step student.

Critical W13 fix: *interleaved* physical rows, corrected native CUDA file.
Never silently substitute the legacy half-split CUDA kernel.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

from prefill_cache_stage5u import CachedCarrierWeights
from quantized_store_stage5v import QuantizedGPUStore


class NativeLibrary5W:
    def __init__(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'corrected Stage-5W CUDA library missing: {path}')
        lib = ctypes.CDLL(str(path))
        lib.inkling5w_workspace_create.argtypes = []
        lib.inkling5w_workspace_create.restype = ctypes.c_void_p
        lib.inkling5w_workspace_destroy.argtypes = [ctypes.c_void_p]
        lib.inkling5w_workspace_destroy.restype = None
        lib.inkling5w_run.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint64] * 3 +
            [ctypes.c_float, ctypes.c_int, ctypes.c_int] + [ctypes.c_uint64] * 11)
        lib.inkling5w_run.restype = ctypes.c_int
        self.lib = lib
        self.workspace = lib.inkling5w_workspace_create()
        if not self.workspace:
            raise RuntimeError('native CUDA MoE workspace creation failed')

    def close(self):
        if self.workspace:
            self.lib.inkling5w_workspace_destroy(self.workspace)
            self.workspace = None

    def run(self, x, router, bias, route_scale, fmt, count, a, b, mapping, output, stream):
        p = lambda t: 0 if t is None else int(t.data_ptr())
        code = self.lib.inkling5w_run(
            self.workspace,p(x),p(router),p(bias),float(route_scale),fmt,count,
            p(a[0]),p(a[1]),p(a[2]),p(a[3]),
            p(b[0]),p(b[1]),p(b[2]),p(b[3]),
            p(mapping),p(output),int(stream))
        if code:
            raise RuntimeError(f'corrected native MoE CUDA launch failed: cudaError={code}')


class NativeQuantizedCarrierWeights(CachedCarrierWeights):
    """Use native quantized MoE; nonexpert projections retain Stage-5U FP32 cache."""
    def __init__(self, reader, device, geometry, *, library_path,
                 nonexpert_mib=2048, quantized_mib=4096, verify_first=False):
        import torch
        super().__init__(reader, device, geometry, nonexpert_mib=nonexpert_mib,
                         expert_mib=0)
        if torch.device(device).type != 'cuda':
            raise ValueError('native MoE requires CUDA')
        self.quant = QuantizedGPUStore(reader, device, geometry, budget_mib=quantized_mib)
        self.native = NativeLibrary5W(library_path)
        self._verify_first = verify_first
        self._verified = False
        self.native_calls = 0
        self.max_verification_error = None

    def moe(self, hidden, layer):
        import torch
        import torch.nn.functional as F
        g = self.geom
        if hidden.shape != (1,1,g.hidden) or not 0 <= layer < 24:
            raise ValueError('native MoE input shape or layer mismatch')
        if (g.hidden,g.intermediate,g.experts,g.top_k)!=(2304,1152,128,8):
            raise ValueError('native CUDA MoE supports only Inkling 2 Mini geometry')
        prefix = f'model.llm.layers.{layer}.mlp.'
        router = self.tensor(prefix+'gate.weight',(128,2304)).contiguous()
        bias = self.tensor(prefix+'gate.bias',(128,)).contiguous()
        scores = torch.sigmoid(F.linear(hidden.float(), router).reshape(128))
        chosen = torch.topk(scores+bias,k=8,sorted=True).indices
        ids = [int(i) for i in chosen.cpu().tolist()]
        parts = self.quant.select(layer,ids)
        # Packed 8-bank view assembled on GPU. Encoded source bytes stay encoded.
        # Next optimization: eliminate this staging copy by contiguous placement.
        a_codes = torch.stack([v.w13_codes for v in parts]).contiguous()
        a_scales = torch.stack([v.w13_scales for v in parts]).contiguous()
        b_codes = torch.stack([v.w2_codes for v in parts]).contiguous()
        b_scales = torch.stack([v.w2_scales for v in parts]).contiguous()
        if layer < 20:
            to_gpu = lambda vals: torch.tensor(vals,device=self.device,dtype=torch.float32)
            a_amax=to_gpu([v.w13_amax for v in parts]); a_den=to_gpu([v.w13_denominator for v in parts])
            b_amax=to_gpu([v.w2_amax for v in parts]); b_den=to_gpu([v.w2_denominator for v in parts])
        else:
            a_amax=a_den=b_amax=b_den=None
        mapping = torch.full((128,),-1,dtype=torch.int32,device=self.device)
        mapping[chosen] = torch.arange(8,dtype=torch.int32,device=self.device)
        out = torch.empty((2304,),dtype=torch.float32,device=self.device)
        fmt=1 if layer<20 else 2
        stream = torch.cuda.current_stream(self.device)
        x = hidden.reshape(-1).contiguous()
        self.native.run(x,router,bias,g.route_scale,fmt,8,
            (a_codes,a_scales,a_amax,a_den),(b_codes,b_scales,b_amax,b_den),mapping,out,stream.cuda_stream)
        # CUDA launch through ctypes is invisible to Torch's stream dependency tracker.
        # Synchronize before temporary staging buffers are released/reused.
        stream.synchronize()
        if not bool(torch.isfinite(out).all()):
            raise ValueError('native quantized MoE produced nonfinite output (selected expert missing or kernel failure)')
        self.native_calls += 1
        if self._verify_first and not self._verified:
            reference, ref_ids = super().moe(hidden,layer)
            max_abs = (reference.reshape(-1)-out).abs().max().item()
            self.max_verification_error = max_abs
            if set(ref_ids) != set(ids) or not torch.allclose(reference.reshape(-1),out,rtol=0.002,atol=0.005):
                raise ValueError(f'native vs corrected Stage5U FP32 mismatch: layer={layer} error={max_abs:.6g}')
            self._verified = True
        return out.reshape(1,1,-1),ids

    def metrics(self):
        return {**super().metrics(),**self.quant.metrics(),
                'native_moe_calls':self.native_calls,
                'native_verified_first':self._verified,
                'native_first_max_abs_error':self.max_verification_error}

    def release(self):
        self.quant.release()
        self.native.close()
        super().release()

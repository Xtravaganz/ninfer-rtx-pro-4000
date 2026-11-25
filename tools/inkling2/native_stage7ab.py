"""Opt-in fast single-token MoE using the existing immutable Stage6G banks."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import math


class FastMoE:
    def __init__(self, native, path):
        self.native = native  # owner of store, must outlive us
        self.lib = C.CDLL(str(Path(path).resolve()))
        self.handle = None
        P, U, I, F = C.c_void_p, C.c_uint64, C.c_int, C.c_float
        for name, args, result in (
            ('create', [], P), ('destroy', [P], None),
            ('run', [P, P, I, U, U, U, F, U, U], I),
            ('routes', [P, U, U, U], I),
            ('reference_routes', [U, U, U, F, U, U, U], I),
        ):
            fn = getattr(self.lib, 'inkling7ab_' + name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.lib.inkling7ab_create()
        if not self.handle:
            raise RuntimeError('Stage7AB workspace creation failed')
        self.calls = 0

    @staticmethod
    def check(code, phase):
        if code:
            raise RuntimeError(f'Stage7AB {phase}: CUDA/result {code}')

    def run(self, x, router, bias, route_scale, layer, out, stream):
        import torch
        if not self.handle or layer not in self.native.loaded or not math.isfinite(route_scale) or route_scale <= 0:
            raise ValueError('Stage7AB requires a live workspace, resident layer and positive route scale')
        for tensor, shape in ((x,(2304,)),(router,(128,2304)),(bias,(128,)),(out,(2304,))):
            if (tuple(tensor.shape)!=shape or tensor.dtype!=torch.float32 or
                    tensor.device!=torch.device('cuda:0') or not tensor.is_contiguous() or tensor.data_ptr()%16):
                raise ValueError(f'Stage7AB expected aligned contiguous cuda:0 FP32 {shape}')
        if stream.device!=x.device or x.data_ptr()==out.data_ptr():
            raise ValueError('Stage7AB stream mismatch or input/output alias')
        self.check(self.lib.inkling7ab_run(self.native.store,self.handle,layer,x.data_ptr(),router.data_ptr(),
                    bias.data_ptr(),route_scale,out.data_ptr(),int(stream.cuda_stream)), 'run')
        self.calls += 1

    def routes(self, x, router, bias, scale, stream):
        """Diagnostic only, no synchronization or readback inside the native operator."""
        import torch
        ids=torch.empty(8,device=x.device,dtype=torch.int32)
        coeff=torch.empty(8,device=x.device,dtype=torch.float32)
        ref_ids=torch.empty_like(ids); ref_coeff=torch.empty_like(coeff)
        self.check(self.lib.inkling7ab_routes(self.handle,ids.data_ptr(),coeff.data_ptr(),int(stream.cuda_stream)), 'routes')
        self.check(self.lib.inkling7ab_reference_routes(x.data_ptr(),router.data_ptr(),bias.data_ptr(),scale,
                    ref_ids.data_ptr(),ref_coeff.data_ptr(),int(stream.cuda_stream)), 'reference_routes')
        return ids,coeff,ref_ids,ref_coeff

    def close(self):
        # Caller synchronizes its stream before releasing this workspace.
        if self.handle:
            self.lib.inkling7ab_destroy(self.handle)
            self.handle=None

"""Opt-in Inkling stages 6G/6H/6I: C++/CUDA-owned fully resident expert banks.

Only MoE operators are native. Source reading, ordinary layers, cache/loop and
chat remain Stage5Q PyTorch; the full native model executor is NOT implemented.

No expert weights are copied to a Torch GPU cache. Every physical layer's 128
experts are uploaded once, then kept immutably on GPU for the process lifetime.
The CUDA router computes top-8 IDs on device; normal execution reads no IDs to
CPU and contains no per-MoE stream synchronization. CUDA completion is enforced
at process shutdown and at explicit parity/perf boundaries only.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from prefill_cache_stage5u import CachedCarrierWeights
from quantized_store_stage5v import expert_geometry, validate_expert_words

E = 128


def plane_sizes(layer: int) -> tuple[int, int, int, int]:
    """Exact source-order bytes for a FULL [128, 2304, K] expert bank."""
    if type(layer) is not int or not 0 <= layer < 24:
        raise ValueError('layer must be in [0,23]')
    nv = layer < 20
    sizes = []
    for projection in ('w13', 'w2'):
        rows, cols, fmt = expert_geometry(layer, projection)
        assert (fmt == 'nvfp4') == nv
        sizes.extend((E * rows * cols // (2 if nv else 1),
                      E * rows * cols // (16 if nv else 32)))
    return tuple(sizes)


def _plane(items, field: str, expected: int) -> bytearray:
    payload = bytearray(expected)
    stride = expected // E
    for e, raw in enumerate(items):
        data = getattr(raw, field)
        if len(data) != stride:
            raise ValueError(f'{field}: expert {e} expected {stride} bytes, got {len(data)}')
        payload[e * stride:(e + 1) * stride] = data
    return payload


def pack_full_bank(layer: int, up, down, *, validate=True):
    """Host plane packing; bytes stay in original source order (no expert GEMV staging).

    Only startup uses this function. The C++ store owns ALL GPU allocations and
    performs synchronous upload; Python Host buffers can be released on return.
    """
    if len(up) != E or len(down) != E:
        raise ValueError('native resident bank requires all 128 physical experts')
    if validate:
        for projection, records in (('w13', up), ('w2', down)):
            for item in records:
                validate_expert_words(layer, projection, item)
    a, b, c, d = plane_sizes(layer)
    codes13 = _plane(up, 'codes', a)
    scales13 = _plane(up, 'scales', b)
    codes2 = _plane(down, 'codes', c)
    scales2 = _plane(down, 'scales', d)
    factors = np.ones((4, E), dtype=np.float32)
    if layer < 20:
        for e, (first, second) in enumerate(zip(up, down)):
            factors[:,e] = (first.amax, first.denominator, second.amax, second.denominator)
    if not np.isfinite(factors).all() or not np.all(factors > 0):
        raise ValueError('invalid original weight scales')
    return (codes13, scales13, codes2, scales2), factors


def _ptr(buffer: bytearray) -> int:
    return ctypes.addressof((ctypes.c_uint8 * len(buffer)).from_buffer(buffer))


class NativeResidentLibrary:
    """Single CUDA library for 6G immutable banks, 6H pointwise MoE, 6I batch MoE."""
    def __init__(self, library_path, *, max_batch_tokens: int = 16):
        if not 1 <= max_batch_tokens <= 128:
            raise ValueError('max_batch_tokens must be in [1,128]')
        path = Path(library_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        lib = ctypes.CDLL(str(path))
        addr = ctypes.c_uint64
        lib.inkling6g_store_create.argtypes = []
        lib.inkling6g_store_create.restype = ctypes.c_void_p
        lib.inkling6g_store_destroy.argtypes = [ctypes.c_void_p]
        lib.inkling6g_store_destroy.restype = None
        lib.inkling6g_upload_layer.argtypes = [ctypes.c_void_p, ctypes.c_int] + [addr]*5
        lib.inkling6g_upload_layer.restype = ctypes.c_int
        lib.inkling6g_resident_bytes.argtypes = [ctypes.c_void_p]
        lib.inkling6g_resident_bytes.restype = addr
        lib.inkling6g_resident_layers.argtypes = [ctypes.c_void_p]
        lib.inkling6g_resident_layers.restype = ctypes.c_int
        lib.inkling6e_workspace_create.argtypes = []
        lib.inkling6e_workspace_create.restype = ctypes.c_void_p
        lib.inkling6e_workspace_destroy.argtypes = [ctypes.c_void_p]
        lib.inkling6e_workspace_destroy.restype = None
        lib.inkling6h_run.argtypes = ([ctypes.c_void_p]*2 + [ctypes.c_int] + [addr]*3 +
                                     [ctypes.c_float] + [addr]*2)
        lib.inkling6h_run.restype = ctypes.c_int
        lib.inkling6i_workspace_create.argtypes = [ctypes.c_int]
        lib.inkling6i_workspace_create.restype = ctypes.c_void_p
        lib.inkling6i_workspace_destroy.argtypes = [ctypes.c_void_p]
        lib.inkling6i_workspace_destroy.restype = None
        lib.inkling6i_run.argtypes = ([ctypes.c_void_p]*2 + [ctypes.c_int] + [addr]*3 +
                                     [ctypes.c_float,ctypes.c_int] + [addr]*2)
        lib.inkling6i_run.restype = ctypes.c_int
        self.lib = lib
        self.store = lib.inkling6g_store_create()
        self.single = lib.inkling6e_workspace_create() if self.store else None
        self.batch = lib.inkling6i_workspace_create(max_batch_tokens) if self.single else None
        if not (self.store and self.single and self.batch):
            self.close()
            raise RuntimeError('Stage 6G/H/I CUDA store/workspace creation failed')
        self.loaded = set()
        self.cold_upload_s = 0.0
        self.cold_pack_s = 0.0
        self.cold_bank_s = 0.0
        self.cold_bank_bytes = 0
        self.calls_single = 0
        self.calls_batch = 0

    def upload(self, layer: int, reader) -> None:
        if layer in self.loaded:
            return
        start = perf_counter()
        all_ids = list(range(E))
        up = reader.bank(layer, 'w13', all_ids)
        down = reader.bank(layer, 'w2', all_ids)
        self.cold_bank_s += perf_counter() - start
        pack_start = perf_counter()
        plane_buffers, scales = pack_full_bank(layer, up, down)
        self.cold_pack_s += perf_counter() - pack_start
        del up, down
        self.cold_bank_bytes += sum(len(x) for x in plane_buffers)
        start = perf_counter()
        code = self.lib.inkling6g_upload_layer(
            self.store, layer, *[_ptr(x) for x in plane_buffers], int(scales.ctypes.data))
        self.cold_upload_s += perf_counter() - start
        if code:
            raise RuntimeError(f'Stage6G failed to upload layer {layer}: CUDA/result {code}; '
                               'full residency needs ~15 GiB expert VRAM, plus ordinary weights')
        self.loaded.add(layer)

    def run(self, hidden, router, bias, route_scale, layer, out, stream) -> None:
        code = self.lib.inkling6h_run(
            self.store, self.single, int(layer),
            hidden.data_ptr(), router.data_ptr(), bias.data_ptr(), float(route_scale),
            out.data_ptr(), int(stream.cuda_stream))
        if code:
            raise RuntimeError(f'Stage6H native MoE failure layer={layer} error={code}')
        self.calls_single += 1

    def run_batch(self, hidden, router, bias, route_scale, layer, out, stream) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != 2304 or tuple(out.shape) != tuple(hidden.shape):
            raise ValueError('Stage6I requires contiguous [tokens, 2304] input and output')
        code = self.lib.inkling6i_run(
            self.batch, self.store, int(layer),
            hidden.data_ptr(), router.data_ptr(), bias.data_ptr(), float(route_scale),
            int(hidden.shape[0]), out.data_ptr(), int(stream.cuda_stream))
        if code:
            raise RuntimeError(f'Stage6I native batch MoE failed: CUDA/result {code}')
        self.calls_batch += 1

    def metrics(self):
        return {'native_resident_layers': self.lib.inkling6g_resident_layers(self.store),
                'native_resident_mib': round(self.lib.inkling6g_resident_bytes(self.store)/1048576,2),
                'native_cold_bank_read_s': round(self.cold_bank_s,3),
                'native_cold_upload_s': round(self.cold_upload_s,3),
                'native_cold_pack_s': round(self.cold_pack_s,3),
                'native_cold_bank_bytes': self.cold_bank_bytes,
                'native_single_moe_calls': self.calls_single,
                'native_batched_moe_calls': self.calls_batch,
                'native_cpu_topk_readbacks': 0,
                'native_per_moe_stream_synchronizations': 0}

    def close(self):
        if getattr(self, 'batch', None):
            self.lib.inkling6i_workspace_destroy(self.batch)
            self.batch = None
        if getattr(self, 'single', None):
            self.lib.inkling6e_workspace_destroy(self.single)
            self.single = None
        if getattr(self, 'store', None):
            self.lib.inkling6g_store_destroy(self.store)
            self.store = None


class ResidentCarrierWeights(CachedCarrierWeights):
    """Only MoE is native; attention/conv/loop stay in original Torch forward."""
    def __init__(self, reader, device, geometry, *, library_path,
                 nonexpert_mib=4096, verify_first=False):
        import torch
        super().__init__(reader, device, geometry, nonexpert_mib=nonexpert_mib, expert_mib=0)
        if self.device.type != 'cuda':
            raise ValueError('native resident MoE requires CUDA')
        # Ensure we create CUDA store on the same GPU as the tensor allocations.
        with torch.cuda.device(self.device):
            self.native = NativeResidentLibrary(library_path)
        self.verify_first = verify_first
        self.verified_first = False
        self.native_first_max_abs_error = None

    def moe(self, hidden, layer):
        import torch
        g = self.geom
        if hidden.shape != (1, 1, 2304) or \
           (g.hidden,g.intermediate,g.experts,g.top_k)!=(2304,1152,128,8) or not 0<=layer<24:
            raise ValueError('unsupported full-resident Inkling geometry')
        # All bank reads and H2D uploads occur only once per PHYSICAL layer.
        # Full prefill still pays these cold costs (reported separately).
        self.native.upload(layer, self.reader)
        p = f'model.llm.layers.{layer}.mlp.'
        router = self.tensor(p + 'gate.weight', (128,2304)).contiguous()
        bias = self.tensor(p + 'gate.bias', (128,)).contiguous()
        x = hidden.reshape(-1).contiguous()
        out = torch.empty((2304,), dtype=torch.float32, device=self.device)
        stream = torch.cuda.current_stream(self.device)
        self.native.run(x, router, bias, g.route_scale, layer, out, stream)
        # Route IDs are not copied to host in production. [] explicitly means
        # unavailable trace; routing and coefficients stay inside native CUDA.
        ids = []
        if self.verify_first and not self.verified_first:
            # Only this opt-in validation path incurs sync/CPU router/ref decode.
            reference, ids = super().moe(hidden, layer)
            error = float((reference.reshape(-1) - out).abs().max().item())
            if not torch.allclose(out,reference.reshape(-1),atol=.005,rtol=.002):
                raise RuntimeError(f'Stage6H vs HF-W13 corrected reference mismatch: {error}')
            self.native_first_max_abs_error = error
            self.verified_first = True
        return out.reshape(1,1,-1), ids

    def metrics(self):
        return {**super().metrics(),**self.native.metrics(),
                'native_verified_first': self.verified_first,
                'native_first_max_abs_error': self.native_first_max_abs_error}

    def release(self):
        # The native store's destructor synchronizes after the last real consumer.
        if hasattr(self,'native'):
            self.native.close()
        super().release()

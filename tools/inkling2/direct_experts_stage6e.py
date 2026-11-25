"""Stage 6E: direct GPU pointer table for encoded Inkling experts.

Opt-in replacement for Stage 5W only. Does not modify the v3 format or
original model checkpoints. CUDA kernels read resident NVFP4/MXFP8 codes
through tables indexed by physical expert ID, avoiding per-token torch.stack.

A GPU->CPU top-k synchronization is STILL required for demand paging. The
model's non-expert operations also remain in the Torch student.
"""
from __future__ import annotations

from collections import defaultdict
import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prefill_cache_stage5u import CachedCarrierWeights
from quantized_store_stage5v import QuantizedGPUStore

EXPERTS = 128


def build_host_tables(items, layer: int, expert_count: int = EXPERTS):
    """Build only small pointer/scale directory words, never concatenate weights.

    Four rows in pointers and factors: W13 codes, W13 scales, W2 codes, W2 scales.
    `valid[eid] == eid` iff the physical expert is completely GPU resident.
    """
    if type(layer) is not int or not 0 <= layer < 24 or expert_count != 128:
        raise ValueError('invalid Inkling 2 Mini layer/expert count')
    pointers = np.zeros((4, expert_count), dtype=np.int64)
    factors = np.ones((4, expert_count), dtype=np.float32)
    valid = np.full((expert_count,), -1, dtype=np.int32)
    for (l, expert), obj in items.items():
        if l != layer:
            continue
        if type(expert) is not int or not 0 <= expert < expert_count:
            raise ValueError('invalid resident expert ID')
        rows = (obj.w13_codes, obj.w13_scales, obj.w2_codes, obj.w2_scales)
        for i, tensor in enumerate(rows):
            address = int(tensor.data_ptr())
            if address <= 0 or address >= 1 << 63:
                raise ValueError('invalid GPU tensor address')
            pointers[i, expert] = address
        factors[0, expert] = float(obj.w13_amax)
        factors[1, expert] = float(obj.w13_denominator)
        factors[2, expert] = float(obj.w2_amax)
        factors[3, expert] = float(obj.w2_denominator)
        if not np.isfinite(factors[:, expert]).all() or np.any(factors[:, expert] <= 0):
            raise ValueError('invalid NVFP4 divisor or amplitude')
        valid[expert] = expert
    return pointers, factors, valid


@dataclass(slots=True)
class DeviceTable:
    pointers: object
    factors: object
    valid: object
    generation: int


class DirectEncodedGPUStore:
    """Wrap Stage5V LRU; track generations of physical layers changed by LRU."""
    def __init__(self, reader, device, geometry, budget_mib=8192):
        import torch
        self.cache = QuantizedGPUStore(reader, device, geometry, budget_mib=budget_mib)
        self.device = torch.device(device)
        self.generation = defaultdict(int)
        self.tables: dict[int, DeviceTable] = {}
        self.table_uploads = 0
        self.table_hits = 0
        self.pointer_bytes = 0
        self.avoided_gather_bytes = 0

    def select_table(self, layer: int, ids: list[int]) -> DeviceTable:
        import torch
        if any(type(i) is not int for i in ids) or len(ids) != 8 or len(set(ids)) != 8:
            raise ValueError('exactly eight unique integer expert IDs required')
        wanted = {(layer, e) for e in ids}
        cache = self.cache
        miss = not wanted.issubset(cache.items.keys())
        before = set(cache.items) if miss else None
        selected = cache.select(layer, ids)
        if miss:
            after = set(cache.items)
            for affected, _ in before.symmetric_difference(after):
                self.generation[affected] += 1
            # Miss was repaired by select; it must have changed at least one key.
            if not after.issuperset(wanted):
                raise RuntimeError('selected expert was evicted during selection')
        version = self.generation[layer]
        existing = self.tables.get(layer)
        if existing is None or existing.generation != version:
            pointers, factors, valid = build_host_tables(cache.items, layer)
            # One small pointer table per physical layer, updated only when
            # this layer gains or loses an expert. NO weight tensor staging.
            table = DeviceTable(
                torch.from_numpy(pointers).to(self.device),
                torch.from_numpy(factors).to(self.device),
                torch.from_numpy(valid).to(self.device),
                version,
            )
            self.tables[layer] = table
            self.table_uploads += 1
            self.pointer_bytes += pointers.nbytes + factors.nbytes + valid.nbytes
        else:
            self.table_hits += 1
            table = existing
        # Estimate eliminated temporary contiguous gather buffer traffic.
        self.avoided_gather_bytes += sum(e.byte_count for e in selected)
        return table

    def metrics(self):
        return {**self.cache.metrics(),
                'pointer_table_uploads': self.table_uploads,
                'pointer_table_hits': self.table_hits,
                'pointer_table_upload_bytes': self.pointer_bytes,
                'avoided_gather_bytes_estimate': self.avoided_gather_bytes,
                'resident_pointer_tables': len(self.tables)}

    def release(self):
        self.tables.clear()
        self.cache.release()


class NativeLibrary6E:
    def __init__(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Stage6E CUDA library missing: {path}')
        lib = ctypes.CDLL(str(path))
        lib.inkling6e_workspace_create.argtypes = []
        lib.inkling6e_workspace_create.restype = ctypes.c_void_p
        lib.inkling6e_workspace_destroy.argtypes = [ctypes.c_void_p]
        lib.inkling6e_workspace_destroy.restype = None
        lib.inkling6e_run.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint64]*3 +
                                      [ctypes.c_float, ctypes.c_int] + [ctypes.c_uint64]*5)
        lib.inkling6e_run.restype = ctypes.c_int
        self.lib = lib
        self.workspace = lib.inkling6e_workspace_create()
        if not self.workspace:
            raise RuntimeError('Stage6E workspace allocation failed')

    def run(self, x, router, bias, scale: float, fmt: int, table, out, stream):
        ptr = lambda x: int(x.data_ptr())
        rc = self.lib.inkling6e_run(self.workspace, ptr(x), ptr(router), ptr(bias),
                                    float(scale), int(fmt), ptr(table.pointers),
                                    ptr(table.factors), ptr(table.valid), ptr(out), int(stream))
        if rc:
            raise RuntimeError(f'inkling6e_run failed: CUDA error {rc}')

    def close(self):
        if self.workspace:
            self.lib.inkling6e_workspace_destroy(self.workspace)
            self.workspace = None


class DirectQuantizedCarrierWeights(CachedCarrierWeights):
    """Stage5W math, but direct encoded expert pointers instead of torch.stack."""
    def __init__(self, reader, device, geometry, *, library_path,
                 nonexpert_mib=4096, quantized_mib=8192, verify_first=False):
        import torch
        super().__init__(reader, device, geometry, nonexpert_mib=nonexpert_mib, expert_mib=0)
        if torch.device(device).type != 'cuda':
            raise ValueError('Stage6E requires CUDA')
        self.direct = DirectEncodedGPUStore(reader, device, geometry, quantized_mib)
        self.native = NativeLibrary6E(library_path)
        self._verify_first = verify_first
        self._verified = False
        self.native_calls = 0
        self.max_verification_error = None

    def moe(self, hidden, layer):
        import torch
        import torch.nn.functional as F
        g = self.geom
        if tuple(hidden.shape) != (1, 1, 2304) or (g.hidden,g.intermediate,g.experts,g.top_k)!=(2304,1152,128,8) or not 0 <= layer < 24:
            raise ValueError('unsupported Stage6E model geometry')
        prefix = f'model.llm.layers.{layer}.mlp.'
        router = self.tensor(prefix + 'gate.weight', (128,2304)).contiguous()
        bias = self.tensor(prefix + 'gate.bias', (128,)).contiguous()
        scores = torch.sigmoid(F.linear(hidden.float(), router).reshape(128))
        chosen = torch.topk(scores + bias, k=8, sorted=True).indices
        ids = [int(i) for i in chosen.cpu().tolist()]
        table = self.direct.select_table(layer, ids)
        out = torch.empty((2304,), dtype=torch.float32, device=self.device)
        fmt = 1 if layer < 20 else 2
        stream = torch.cuda.current_stream(self.device)
        x = hidden.reshape(-1).contiguous()
        self.native.run(x, router, bias, g.route_scale, fmt, table, out, stream.cuda_stream)
        # Keep table tensors + LRU evictions safe while native CUDA consumes pointers.
        # A true fully native executor must eliminate this per-layer synchronization.
        stream.synchronize()
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError('non-resident expert or nonfinite CUDA result')
        self.native_calls += 1
        if self._verify_first and not self._verified:
            reference, ref_ids = super().moe(hidden, layer)
            max_abs = float((reference.reshape(-1) - out).abs().max().item())
            self.max_verification_error = max_abs
            if set(ref_ids) != set(ids) or not torch.allclose(
                reference.reshape(-1), out, rtol=0.002, atol=0.005):
                raise ValueError(f'Stage6E CUDA vs corrected W13 FP32 mismatch: layer={layer} error={max_abs}')
            self._verified = True
        return out.reshape(1, 1, -1), ids

    def metrics(self):
        return {**super().metrics(), **self.direct.metrics(),
                'native_moe_calls': self.native_calls,
                'native_verified_first': self._verified,
                'native_first_max_abs_error': self.max_verification_error}

    def release(self):
        self.direct.release()
        self.native.close()
        super().release()

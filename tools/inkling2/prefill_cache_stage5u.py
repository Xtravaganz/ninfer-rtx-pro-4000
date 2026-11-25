"""Opt-in Stage 5U: bounded GPU FP32 weight caches for the *existing* 5Q student.

- Keeps non-expert weights across layer visits and tokens (unlike 5H.clear()).
- Reuses decoded, selected real NVFP4/MXFP8 experts by (physical layer, projection, id).
- Preserves the HF W13 row order: gate=raw[0::2], up=raw[1::2].
- Does NOT claim HF parity, batch prefill, or a native NInfer runtime.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from time import perf_counter

import torch
import torch.nn.functional as F

from real_sliding_stage5h import TorchCarrierWeights, decode_expert

_MIB = 1024 * 1024


def _mib_budget(value: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 16384:
        raise ValueError(f'{label} must be a whole number of MiB in [0,16384]')
    return value * _MIB


def _tensor_size(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


class CachedCarrierWeights(TorchCarrierWeights):
    """Bounded resident FP32 cache, opt-in replacement for TorchCarrierWeights.

    The cache stores *decoded* expert matrices, hence its budget is **not** the
    14.6 GiB quantized disk size. Conservative defaults fit typical 24-GiB GPUs,
    but CUDA workspaces/other processes still need VRAM. All eviction is LRU.
    """

    def __init__(self, reader, device, geometry, *, nonexpert_mib: int = 2048,
                 expert_mib: int = 4096):
        super().__init__(reader, device, geometry)
        self.nonexpert_limit = _mib_budget(nonexpert_mib, 'nonexpert cache')
        self.expert_limit = _mib_budget(expert_mib, 'expert cache')
        self._cache = OrderedDict()
        self._expert_cache: OrderedDict[tuple[int, str, int], torch.Tensor] = OrderedDict()
        self._nonexpert_bytes = 0
        self._expert_bytes = 0
        self.counters = {
            'nonexpert_hits': 0, 'nonexpert_misses': 0,
            'expert_hits': 0, 'expert_misses': 0,
            'nonexpert_evictions': 0, 'expert_evictions': 0,
            'bank_read_calls': 0, 'bank_read_bytes': 0,
            'nonexpert_read_bytes': 0,
            'bank_read_seconds': 0.0, 'expert_decode_dispatch_seconds': 0.0,
            'clear_requests_ignored': 0,
        }

    def clear(self):
        """Existing 5Q calls clear() after every visit; retain the bounded LRU."""
        self.counters['clear_requests_ignored'] += 1

    def release(self):
        """Explicitly free our references without modifying the 5Q KV/conv cache."""
        self._cache.clear()
        self._expert_cache.clear()
        self._nonexpert_bytes = 0
        self._expert_bytes = 0

    def _put_nonexpert(self, key: str, value: torch.Tensor):
        size = _tensor_size(value)
        if size > self.nonexpert_limit:
            return
        while self._cache and self._nonexpert_bytes + size > self.nonexpert_limit:
            _, old = self._cache.popitem(last=False)
            self._nonexpert_bytes -= _tensor_size(old)
            self.counters['nonexpert_evictions'] += 1
        self._cache[key] = value
        self._nonexpert_bytes += size

    def _put_expert(self, key: tuple[int, str, int], value: torch.Tensor):
        size = _tensor_size(value)
        if size > self.expert_limit:
            return
        while self._expert_cache and self._expert_bytes + size > self.expert_limit:
            _, old = self._expert_cache.popitem(last=False)
            self._expert_bytes -= _tensor_size(old)
            self.counters['expert_evictions'] += 1
        self._expert_cache[key] = value
        self._expert_bytes += size

    def tensor(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        found = self._cache.get(name)
        if found is not None:
            if tuple(found.shape) != tuple(shape):
                raise ValueError(f'{name}: cache geometry mismatch')
            self.counters['nonexpert_hits'] += 1
            self._cache.move_to_end(name)
            return found
        self.counters['nonexpert_misses'] += 1
        raw, fmt, dims = self.reader.nonexpert_tensor(name)
        if tuple(dims) != tuple(shape) or fmt not in ('bf16', 'fp32'):
            raise ValueError(f'{name}: tensor format/shape mismatch; expected {shape}, got {fmt}/{dims}')
        self.counters['nonexpert_read_bytes'] += len(raw)
        dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32
        value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(dims)
        result = value.to(device=self.device, dtype=torch.float32)
        self._put_nonexpert(name, result)
        return result

    def conv(self, name: str, channels: int) -> torch.Tensor:
        # In 5H, conv() reads the entire source tensor merely to inspect its
        # shape and then tensor() reads it AGAIN. Use v3 directory metadata.
        obj_reader = getattr(self.reader, 'nonexpert', None)
        if obj_reader is None or not hasattr(obj_reader, 'object'):
            # Synthetic/fake reader without directory: preserve 5H compatibility.
            return super().conv(name, channels)
        obj = obj_reader.object('hf/' + name)
        shape = tuple(obj.shape)
        kernel = self.geom.conv_kernel
        if shape not in ((channels, kernel), (channels, 1, kernel)):
            raise ValueError(f'{name}: unsupported convolution shape {shape}')
        return self.tensor(name, shape).reshape(channels, kernel)

    def _selected_experts(self, layer: int, projection: str,
                          ids: list[int]) -> list[torch.Tensor]:
        g = self.geom
        if projection not in ('w13','w2') or type(layer) is not int or not 0 <= layer < 24:
            raise ValueError('invalid expert projection or physical layer')
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < g.experts for i in ids):
            raise ValueError('invalid expert IDs')
        fmt = 'nvfp4' if layer < 20 else 'mxfp8'
        rows = 2 * g.intermediate if projection == 'w13' else g.hidden
        cols = g.hidden if projection == 'w13' else g.intermediate
        results: dict[int, torch.Tensor] = {}
        missing = []
        for e in ids:
            key = (layer, projection, e)
            tensor = self._expert_cache.get(key)
            if tensor is not None:
                if tuple(tensor.shape) != (rows, cols):
                    raise ValueError('cached expert shape mismatch')
                self._expert_cache.move_to_end(key)
                results[e] = tensor
                self.counters['expert_hits'] += 1
            else:
                missing.append(e)
                self.counters['expert_misses'] += 1
        if missing:
            start = perf_counter()
            raw = self.reader.bank(layer, projection, missing)
            self.counters['bank_read_seconds'] += perf_counter() - start
            self.counters['bank_read_calls'] += 1
            if len(raw) != len(missing):
                raise ValueError('selected expert bank returned missing weights')
            for e, words in zip(missing, raw):
                self.counters['bank_read_bytes'] += len(words.codes) + len(words.scales)
                begin = perf_counter()
                value = decode_expert(words, rows=rows, cols=cols, fmt=fmt, device=self.device)
                self.counters['expert_decode_dispatch_seconds'] += perf_counter() - begin
                results[e] = value
                self._put_expert((layer, projection, e), value)
        return [results[e] for e in ids]

    def moe(self, hidden: torch.Tensor, layer: int) -> tuple[torch.Tensor, list[int]]:
        """Same FP32 arithmetic and *corrected* W13 partition as Stage 5S/T."""
        g = self.geom
        if hidden.shape != (1, 1, g.hidden) or type(layer) is not int or not 0 <= layer < 24:
            raise ValueError('MoE expects one hidden token and physical layer 0..23')
        p = f'model.llm.layers.{layer}.mlp.'
        router = self.tensor(p + 'gate.weight', (g.experts, g.hidden))
        bias = self.tensor(p + 'gate.bias', (g.experts,))
        scores = torch.sigmoid(F.linear(hidden.float(), router).reshape(g.experts))
        ids_tensor = torch.topk(scores + bias, k=g.top_k, sorted=True).indices
        ids = [int(e) for e in ids_tensor.cpu().tolist()]
        multipliers = scores[ids_tensor] / (scores[ids_tensor].sum() + 1e-20) * g.route_scale
        w13 = self._selected_experts(layer, 'w13', ids)
        w2 = self._selected_experts(layer, 'w2', ids)
        out = torch.zeros(g.hidden, dtype=torch.float32, device=self.device)
        x = hidden.float().reshape(g.hidden)
        for i in range(g.top_k):
            projected = w13[i] @ x
            # Official Inkling2Loop checkpoint conversion: Interleave(dim=1).
            gate, up = projected[0::2], projected[1::2]
            activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate) *
                         25 * torch.tanh(up / 25))
            down = w2[i] @ activated
            out += multipliers[i] * down
        if not bool(torch.isfinite(out).all()):
            raise ValueError('nonfinite cached MoE output')
        return out.reshape(1, 1, -1), ids

    def metrics(self) -> dict:
        return {**self.counters,
                'resident_nonexpert_mib': round(self._nonexpert_bytes / _MIB, 2),
                'resident_expert_mib': round(self._expert_bytes / _MIB, 2),
                'nonexpert_cache_limit_mib': self.nonexpert_limit // _MIB,
                'expert_cache_limit_mib': self.expert_limit // _MIB,
                'resident_nonexpert_tensors': len(self._cache),
                'resident_expert_matrices': len(self._expert_cache)}

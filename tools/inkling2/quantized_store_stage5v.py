"""Stage 5V: bounded GPU-resident *encoded* Inkling expert cache.

Weights remain source-order NVFP4/MXFP8, no full FP32 materialization.
The cache is opt-in and is used only by the 5W native CUDA MoE bridge.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from time import perf_counter


def expert_geometry(layer: int, projection: str, hidden: int = 2304,
                    intermediate: int = 1152) -> tuple[int, int, str]:
    if type(layer) is not int or not 0 <= layer < 24:
        raise ValueError('physical layer must be 0..23')
    if projection not in ('w13', 'w2'):
        raise ValueError('expert projection must be w13 or w2')
    if hidden <= 0 or intermediate <= 0:
        raise ValueError('positive dimensions required')
    return (2 * intermediate if projection == 'w13' else hidden,
            hidden if projection == 'w13' else intermediate,
            'nvfp4' if layer < 20 else 'mxfp8')


def validate_expert_words(layer: int, projection: str, raw,
                          hidden: int = 2304, intermediate: int = 1152) -> int:
    rows, cols, fmt = expert_geometry(layer, projection, hidden, intermediate)
    code_bytes = rows * cols // (2 if fmt == 'nvfp4' else 1)
    scale_bytes = rows * cols // (16 if fmt == 'nvfp4' else 32)
    if len(raw.codes) != code_bytes or len(raw.scales) != scale_bytes:
        raise ValueError('quantized expert geometry mismatch')
    # Vectorized validation matters: one original MXFP8 expert has megabytes
    # of code words. Python per-byte iteration would dominate prefill time.
    try:
        import numpy as np
        scales = np.frombuffer(raw.scales, dtype=np.uint8)
        if fmt == 'nvfp4':
            invalid = bool(np.any((scales & 0x80) != 0) or np.any(scales == 0x7f))
        else:
            codes = np.frombuffer(raw.codes, dtype=np.uint8)
            invalid = bool(np.any(scales == 0xff) or np.any((codes & 0x7f) == 0x7f))
    except ImportError:
        invalid = (any(b & 0x80 or b == 0x7f for b in raw.scales) if fmt == 'nvfp4'
                   else any(b == 0xff for b in raw.scales) or
                        any((b & 0x7f) == 0x7f for b in raw.codes))
    if fmt == 'nvfp4':
        invalid = invalid or not all(math.isfinite(x) and x > 0 for x in (raw.amax, raw.denominator))
    if invalid:
        raise ValueError('invalid quantized expert words or FP32 scale')
    return code_bytes + scale_bytes + (8 if fmt == 'nvfp4' else 0)


@dataclass(slots=True)
class ResidentExpert:
    w13_codes: object
    w13_scales: object
    w2_codes: object
    w2_scales: object
    w13_amax: float
    w13_denominator: float
    w2_amax: float
    w2_denominator: float
    byte_count: int


class QuantizedGPUStore:
    """LRU of source-order GPU uint8 matrices, grouped by (layer, expert).

    Both projections are admitted or evicted as a single atomic entry. The
    caller keeps returned references alive until the active CUDA stream finishes.
    """
    def __init__(self, reader, device, geometry, *, budget_mib: int = 4096):
        import torch
        if type(budget_mib) is not int or not 1 <= budget_mib <= 16384:
            raise ValueError('quantized cache budget must be 1..16384 MiB')
        self.reader, self.device, self.geometry = reader, torch.device(device), geometry
        if self.device.type != 'cuda':
            raise ValueError('GPU quantized store requires CUDA')
        self.budget_bytes = budget_mib * 1048576
        self.items: OrderedDict[tuple[int, int], ResidentExpert] = OrderedDict()
        self.used_bytes = 0
        self.stats = dict(hits=0, misses=0, evictions=0, bank_reads=0,
                          bank_read_bytes=0, uploads=0, read_seconds=0.0)

    def _upload(self, words, rows: int, cols: int, fmt: str):
        import torch
        divisor = 2 if fmt == 'nvfp4' else 1
        group = 16 if fmt == 'nvfp4' else 32
        codes = torch.frombuffer(bytearray(words.codes), dtype=torch.uint8).reshape(rows, cols // divisor)
        scales = torch.frombuffer(bytearray(words.scales), dtype=torch.uint8).reshape(rows, cols // group)
        self.stats['uploads'] += 2
        return codes.to(self.device), scales.to(self.device)

    def select(self, layer: int, ids: list[int]) -> list[ResidentExpert]:
        if (type(layer) is not int or not 0 <= layer < 24 or not ids or
            len(set(ids)) != len(ids) or any(type(e) is not int or not 0 <= e < self.geometry.experts for e in ids)):
            raise ValueError('invalid requested expert IDs')
        missing = []
        for eid in ids:
            key = (layer, eid)
            if key in self.items:
                self.items.move_to_end(key)
                self.stats['hits'] += 1
            else:
                self.stats['misses'] += 1
                missing.append(eid)
        if missing:
            start = perf_counter()
            up = self.reader.bank(layer, 'w13', missing)
            down = self.reader.bank(layer, 'w2', missing)
            self.stats['read_seconds'] += perf_counter() - start
            self.stats['bank_reads'] += 2
            if len(up) != len(missing) or len(down) != len(missing):
                raise ValueError('bank reader returned incomplete expert selection')
            for eid, a, b in zip(missing, up, down):
                na, ka, fmt = expert_geometry(layer, 'w13', self.geometry.hidden, self.geometry.intermediate)
                nb, kb, _ = expert_geometry(layer, 'w2', self.geometry.hidden, self.geometry.intermediate)
                size = (validate_expert_words(layer, 'w13', a, self.geometry.hidden, self.geometry.intermediate)
                        + validate_expert_words(layer, 'w2', b, self.geometry.hidden, self.geometry.intermediate))
                if size > self.budget_bytes:
                    raise ValueError('one expert pair exceeds the quantized cache budget')
                # Do not evict any expert selected for the *current* call.
                protected = {(layer, requested) for requested in ids}
                while self.used_bytes + size > self.budget_bytes:
                    victim = next((key for key in self.items if key not in protected), None)
                    if victim is None:
                        raise ValueError('selected top-k encoded experts do not fit quantized cache budget')
                    old = self.items.pop(victim)
                    self.used_bytes -= old.byte_count
                    self.stats['evictions'] += 1
                ca, sa = self._upload(a, na, ka, fmt)
                cb, sb = self._upload(b, nb, kb, fmt)
                self.items[(layer,eid)] = ResidentExpert(ca,sa,cb,sb,
                    float(a.amax),float(a.denominator),float(b.amax),float(b.denominator),size)
                self.used_bytes += size
                self.stats['bank_read_bytes'] += size
        return [self.items[(layer,e)] for e in ids]

    def metrics(self):
        return {**self.stats, 'quantized_resident_mib': round(self.used_bytes/1048576,2),
                'quantized_cache_mib':self.budget_bytes//1048576,
                'quantized_resident_experts':len(self.items)}

    def release(self):
        self.items.clear()
        self.used_bytes = 0

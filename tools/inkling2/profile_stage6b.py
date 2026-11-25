"""Stage 6B opt-in detailed, synchronization-aware profiler for Stage 5W.

Do not confuse wall timers with exclusive device time: nested spans overlap and
CUDA-event timings only measure GPU execution between recorded stream events.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from time import perf_counter


@dataclass
class Metric:
    calls: int = 0
    inclusive_s: float = 0.0
    exclusive_s: float = 0.0


class SpanProfiler:
    def __init__(self, *, synchronize_cuda: bool = False):
        self.sync = synchronize_cuda
        self.metrics = defaultdict(Metric)
        self.stack: list[list] = []
        self.notes = []

    @contextmanager
    def span(self, label, device=None):
        if self.sync and device is not None and getattr(device, 'type', None) == 'cuda':
            import torch
            torch.cuda.synchronize(device)
        start = perf_counter()
        frame = [str(label), start, 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            if self.sync and device is not None and getattr(device, 'type', None) == 'cuda':
                import torch
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
            if self.stack[-1] is not frame:
                raise RuntimeError('profiler stack corrupted')
            self.stack.pop()
            metric = self.metrics[label]
            metric.calls += 1
            metric.inclusive_s += elapsed
            metric.exclusive_s += max(0.0, elapsed - frame[2])
            if self.stack:
                self.stack[-1][2] += elapsed

    def wrap(self, obj, method, label, *, device_attr=None):
        original = getattr(obj, method)
        @wraps(original)
        def measured(*args, **kwargs):
            device = (getattr(obj, device_attr) if device_attr else None)
            with self.span(label, device=device):
                return original(*args, **kwargs)
        setattr(obj, method, measured)
        return original

    def result(self):
        return {
            'timing_mode': 'CUDA synchronized; intrusive' if self.sync else 'host wall; async launches may be undercounted',
            'sections': {name: {'calls': m.calls, 'inclusive_s': round(m.inclusive_s, 6),
                                'exclusive_s': round(m.exclusive_s, 6)}
                         for name, m in sorted(self.metrics.items())},
            'caution': 'Nested inclusive spans overlap. Host wall times and CUDA kernel times are not interchangeable.'
        }


class PatchScope:
    def __init__(self):
        self.patches = []

    def patch(self, obj, name, replacement):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def close(self):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)
        self.patches.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

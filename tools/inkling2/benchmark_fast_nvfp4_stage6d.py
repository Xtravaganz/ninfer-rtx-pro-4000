"""Microbenchmark the exact NVFP4 scale permutation; not full model timing."""
from __future__ import annotations
from time import perf_counter
import argparse
import random
from ninfer_artifact_bridge import unswizzle_nvfp4_128_rows


def slow(s, cols):
    stride = cols // 16
    out = bytearray(len(s))
    cursor = 0
    for tile in range(cols // 64):
        for row in range(32):
            for group in range(4):
                offset = (group * 32 + row) * stride + tile * 4
                out[offset:offset + 4] = s[cursor:cursor + 4]
                cursor += 4
    return bytes(out)


def bench(fn, payload, cols, reps=3):
    start = perf_counter()
    for _ in range(reps):
        for _ in range(18):
            fn(payload, cols)
    return 1000 * (perf_counter()-start) / reps


if __name__ == '__main__':
    for k in (1152,2304):
        payload = random.Random(k).randbytes(128 * k//16)
        assert slow(payload,k) == unswizzle_nvfp4_128_rows(payload,k)
        old=bench(slow,payload,k)
        new=bench(unswizzle_nvfp4_128_rows,payload,k)
        print(f'NVFP4 K={k}: 18 tiles/expert old={old:.3f}ms fast={new:.3f}ms '
              f'speedup={old/new:.1f}x (microbenchmark only)')

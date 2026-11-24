#!/usr/bin/env python3
"""PyTorch benchmark for Stage-3B A/B reference (GPU-resident tensors).

Compare with the standalone native binary on the same GPU. Timing is two
separately measured processes, not a controlled simultaneous profiler run.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=2304)
    parser.add_argument('--cols', type=int, default=2304)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 32 or args.cols % 32 != 0 or not 1 <= args.iterations <= 5000:
        parser.error('rows>=1, cols a positive multiple of 32, iterations in [1,5000] required')
    import torch
    from mxfp8_reference import synthetic_matrix

    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; run with --gpus device=1')
    matrix = synthetic_matrix(args.rows, args.cols)
    d = torch.device('cuda:0')
    w = torch.frombuffer(bytearray(matrix.codes), dtype=torch.uint8).reshape(args.rows, args.cols).to(d)
    s = torch.frombuffer(bytearray(matrix.scales), dtype=torch.uint8).reshape(args.rows, args.cols//32).to(d)
    x = torch.tensor([(i%41-20)*0.03125 for i in range(args.cols)], dtype=torch.float32, device=d)

    def dequant():
        return w.view(torch.float8_e4m3fn).float() * torch.exp2(
            (s.to(torch.int16) - 127).to(torch.float32)).repeat_interleave(32, -1)

    matrix_fp32 = dequant()
    torch.cuda.synchronize(d)

    def bench(fn):
        for _ in range(20):
            fn()
        torch.cuda.synchronize(d)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            fn()
        stop.record()
        stop.synchronize()
        return 1000.0*start.elapsed_time(stop)/args.iterations

    print(f'PyTorch GPU {torch.cuda.get_device_name(0)}, shape={args.rows}x{args.cols}')
    print(f'BENCH torch FP32 GEMV (pre-dequantized): {bench(lambda: torch.mv(matrix_fp32, x)):.3f} us/call')
    print(f'BENCH torch resident dequant + GEMV:    {bench(lambda: torch.mv(dequant(), x)):.3f} us/call')
    print('PyTorch FP32 dequant is NOT a native MXFP8 Tensor-Core path.')

if __name__ == '__main__':
    main()

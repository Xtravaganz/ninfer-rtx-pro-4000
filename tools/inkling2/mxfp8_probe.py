#!/usr/bin/env python3
"""Stage-3A MXFP8 numerical probe: CPU oracle vs PyTorch CPU/CUDA dequant+GEMV.

Examples:
  python3 tools/inkling2/mxfp8_probe.py --device cpu
  python3 tools/inkling2/mxfp8_probe.py --device cuda:0
  python3 tools/inkling2/mxfp8_probe.py --device cuda:0 --artifact /data/experts.ninfer \\
    --layer 20 --projection w13 --expert 0 --first 0 --count 8

No production kernels: PyTorch materializes a selected small weight slice.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
import sys

from mxfp8_reference import (
    Matrix, cpu_gemv, read_stage2a_rows, read_stage2b_rows,
    synthetic_matrix, torch_decode, torch_gemv,
)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--bank', type=Path, help='Stage-2A MXFP8 .i2bank file')
    inputs.add_argument('--artifact', type=Path, help='Stage-2B .ninfer expert-only carrier')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0')
    parser.add_argument('--layer', type=int, default=20, help='Stage-2B layer 20..23')
    parser.add_argument('--projection', choices=('w13', 'w2'), default='w13')
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--first', type=int, default=0, help='first output row')
    parser.add_argument('--count', type=int, default=8, help='rows to decode, max 128')
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 128:
        parser.error('count must be 1..128 to bound numerical oracle memory')
    if args.bank:
        matrix = read_stage2a_rows(args.bank, expert_id=args.expert, first=args.first, count=args.count)
        source = 'Stage-2A bank'
    elif args.artifact:
        matrix = read_stage2b_rows(args.artifact, layer=args.layer, projection=args.projection,
                                   expert_id=args.expert, first=args.first, count=args.count)
        source = 'Stage-2B NInfer v3 carrier'
    else:
        matrix = synthetic_matrix()
        source = 'deterministic synthetic MXFP8 matrix'
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: rerun Docker with --gpus and choose --device cuda:0')
    # GPU GEMV consumes float32 activations; make the independent CPU oracle use
    # the SAME rounded float32 input values rather than arbitrary Python f64.
    x = [struct.unpack('<f', struct.pack('<f', math.sin((i + 1) * 0.13) / (1 + i % 3)))[0]
         for i in range(matrix.cols)]
    decoded = torch_decode(matrix, device=args.device)
    # Small selected slice: exact FP32 representability for these ordinary scales.
    # Extreme E8M0 0x00 is checked independently in scalar C++/Python tests.
    numeric = decoded.detach().cpu().reshape(-1).tolist()  # One GPU->CPU transfer, no per-element sync
    expected_words = [matrix.at(r, c) for r in range(matrix.rows) for c in range(matrix.cols)]
    errors = [abs(a - b) for a, b in zip(expected_words, numeric)]
    reference = cpu_gemv(matrix, x)
    actual = torch_gemv(matrix, x, device=args.device).cpu().tolist()
    gemv_err = max(abs(a - b) for a, b in zip(reference, actual))
    gemv_allowed = max(1e-5, 2e-6 * max(abs(value) for value in reference))
    print(f'Source: {source}; matrix={matrix.rows}x{matrix.cols}; device={args.device}')
    print(f'MXFP8 E4M3FN/G32/UE8M0 decode max abs error: {max(errors):.9g}')
    print(f'MXFP8 GEMV max abs error: {gemv_err:.9g}; threshold: {gemv_allowed:.9g}')
    if any(error > (1e-6 + 1e-6 * abs(expected))
           for expected, error in zip(expected_words, errors)):
        raise RuntimeError('decoded GPU/CPU torch values differ from independent scalar oracle')
    if gemv_err > gemv_allowed:
        raise RuntimeError('GEMV numerical discrepancy exceeds threshold')
    print('OK Stage 3A numerical reference (NOT a production GPU kernel)')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(run())
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

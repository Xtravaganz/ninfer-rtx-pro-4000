#!/usr/bin/env python3
"""Pack stage-2A expert banks into a NInfer v3 expert-only carrier (NOT runnable).

Metadata-only plan, no torch and no weight downloads:
  python3 tools/inkling2/pack_ninfer_artifact.py --plan \
    --config /tmp/inkling2-config.json --index /tmp/inkling2-index.json \
    --header /tmp/inkling2-header.json

Pack existing Stage-2A banks (possibly a subset):
  python3 tools/inkling2/pack_ninfer_artifact.py \
    --source /data/inkling2-native-banks \
    --config /data/Inkling-2-Mini/config.json \
    --output /data/inkling2-experts-only.ninfer

Verify original codes/scales are preserved:
  python3 tools/inkling2/pack_ninfer_artifact.py --verify \
    --source /data/inkling2-native-banks --output /data/inkling2-experts-only.ninfer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from check_checkpoint import validate_config, validate_header, validate_index
from native_experts import MiniGeometry, format_for_layer, FMT_NVFP4, parse_selection
from export_experts import expected_bank_size
from ninfer_artifact_bridge import (
    load_manifest, pack_banks, validate_selection, verify_artifact,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, help='Original HF config.json')
    p.add_argument('--index', type=Path, help='Optional original HF safetensors index for plan')
    p.add_argument('--header', type=Path, help='Optional original safetensors header, plan only')
    p.add_argument('--source', type=Path, help='Stage-2A directory with .i2bank + manifest.json')
    p.add_argument('--output', type=Path, help='NInfer v3 expert-only carrier path')
    p.add_argument('--layers', default='all', help='Plan only: all or selection 0,6,20-23')
    p.add_argument('--experts', default='all', help='Plan only: all or selection 0-7')
    p.add_argument('--chunk-mib', type=int, default=4, help='Bounded byte reads for export/verify')
    p.add_argument('--require-complete', action='store_true', help='Require all 48 complete banks')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Only metadata; no weights required')
    mode.add_argument('--verify', action='store_true', help='Compare NInfer container against Stage-2A input')
    args = p.parse_args(argv)
    if not 1 <= args.chunk_mib <= 128:
        p.error('--chunk-mib must be 1..128')
    chunk_bytes = args.chunk_mib * (1 << 20)
    if args.plan:
        if args.config is None:
            p.error('--plan requires --config')
        if not args.index and not args.source:
            p.error('--plan requires --index or --source')
        config = json.loads(args.config.read_text())
        validate_config(config)
        if args.index:
            validate_index(json.loads(args.index.read_text()))
        if args.header:
            validate_header(json.loads(args.header.read_text()))
            print('Official HF tensor dtype/shape contract: OK')
        geometry = MiniGeometry()
        if args.source:
            banks = load_manifest(args.source)
            validate_selection(banks, full_required=args.require_complete)
            native = sum(b.fmt == FMT_NVFP4 for b in banks)
            mxfp8 = len(banks) - native
            # Estimate payload includes div words and separate FP32 source sidecars.
            payload = sum(len(b.ids) * (b.code_stride+b.scale_stride+(4+8 if b.fmt == FMT_NVFP4 else 0)+4)
                          for b in banks)
            print(f'PLAN: {len(banks)} staged banks; {native} typed NVFP4, {mxfp8} raw MXFP8; '
                  f'payload ~{payload/(1<<30):.3f} GiB')
        else:
            layers = parse_selection(args.layers, 24)
            experts = parse_selection(args.experts, 128)
            if args.require_complete and (len(layers) != 24 or len(experts) != 128):
                raise ValueError('complete plan requires all 24 layers and 128 experts')
            payload = sum(expected_bank_size(layer, proj, len(experts))
                          for layer in layers for proj in ('w13', 'w2'))
            print(f'PLAN: {len(layers)} layers × 2 projections × {len(experts)} experts; '
                  f'Stage-2A input ~{payload/(1<<30):.3f} GiB')
        print('NInfer v3 output is an expert-only carrier, not an inference-ready model.')
        print('MXFP8 G32/UE8M0 remains opaque raw_bytes_v1 resources; no GPU kernel implied.')
        return 0
    if args.source is None or args.output is None:
        p.error('export/verify require --source and --output')
    if args.verify:
        result = verify_artifact(args.source, args.output, chunk_bytes=chunk_bytes)
        print(f"VERIFY OK: {result['banks']} banks; {result['native_nvfp4']} native NVFP4; "
              f"{result['mxfp8_resources']} MXFP8 resource banks")
        return 0
    if args.config is None:
        p.error('export requires --config')
    config = json.loads(args.config.read_text())
    result = pack_banks(args.source, args.output, config, chunk_bytes=chunk_bytes,
                        full_required=args.require_complete)
    print(f"PACK OK: {result['banks']} banks; {result['native_nvfp4_banks']} native NVFP4; "
          f"{result['opaque_mxfp8_banks']} MXFP8 resources")
    print(f"NInfer v3 expert-only artifact: {result['artifact']}")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)

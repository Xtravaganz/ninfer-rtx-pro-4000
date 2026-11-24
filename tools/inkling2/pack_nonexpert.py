#!/usr/bin/env python3
"""Stage 4A: main-checkpoint Inkling2 non-expert weights -> NInfer v3 (NOT runnable).

Source is the existing Hugging Face cache snapshot (no downloads). Two separate
NInfer-v3 carriers are required so far: Stage2B expert-only and this remainder.

Examples:
  python3 tools/inkling2/pack_nonexpert.py --source "$MODEL" --plan
  python3 tools/inkling2/pack_nonexpert.py --source "$MODEL" \
    --include 'model.llm.layers.0.mlp.gate.*' --output "$OUT/rest-smoke.ninfer"
  python3 tools/inkling2/pack_nonexpert.py --source "$MODEL" \
    --verify --output "$OUT/rest-smoke.ninfer"
  python3 tools/inkling2/pack_nonexpert.py --source "$MODEL" \
    --require-complete --output "$OUT/rest-full.ninfer"
  python3 tools/inkling2/pack_nonexpert.py --source "$MODEL" \
    --verify --require-complete --output "$OUT/rest-full.ninfer"

--plan reads index and shard header only; no tensor bytes or artifact writes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Direct CLI execution (python3 tools/inkling2/...) has sys.path[0] in
# tools/inkling2, not the Git repository root. Avoid the Stage-2B PYTHONPATH bug.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonexpert_artifact import Checkpoint, describe_plan, pack_remaining, verify_remaining


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path,
                   help="local HF snapshot folder with config, index, and safetensors")
    p.add_argument("--output", type=Path, help="destination *.ninfer path")
    p.add_argument("--include", action="append", default=[], metavar="GLOB",
                   help="select a subset of non-expert tensor names; repeatable")
    p.add_argument("--require-complete", action="store_true",
                   help="disallow selected subsets; require every remaining main tensor")
    p.add_argument("--chunk-mib", type=int, default=4, metavar="N",
                   help="stream/read comparison chunk size, 1..64 MiB")
    p.add_argument("--max-file-mib", type=int, metavar="N",
                   help="optional artifact shard max file size (MiB)")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="metadata-only planning")
    action.add_argument("--verify", action="store_true", help="byte-verify existing artifact")
    args = p.parse_args(argv)
    if not 1 <= args.chunk_mib <= 64:
        p.error("--chunk-mib must be 1..64")
    if args.max_file_mib is not None and args.max_file_mib < 1:
        p.error("--max-file-mib must be positive")
    if args.require_complete and args.include:
        p.error("--require-complete cannot be combined with --include")
    if args.plan and args.output is not None:
        p.error("--plan does not take --output")
    if not args.plan and args.output is None:
        p.error("pack or --verify requires --output")
    with Checkpoint(args.source) as checkpoint:
        checkpoint.validate_inkling()
        if args.plan:
            spans = checkpoint.select(tuple(args.include))
            result = describe_plan(spans)
            print(f"HF main checkpoint: {len(checkpoint.spans)} indexed tensors; "
                  f"{len(checkpoint.expert_names)} expert source tensors; "
                  f"{len(checkpoint.remaining_names)} non-expert tensors")
            print(f"STAGE4A PLAN: {result['tensors']} tensors; "
                  f"{result['payload_bytes'] / (1 << 30):.3f} GiB NInfer payload; "
                  f"native={sum(v for k, v in result['formats'].items() if k != 'raw')} "
                  f"raw_resources={result['formats']['raw']}")
            print("PLAN ONLY: header/index checked; no tensor payload read")
            print("MTP checkpoint is a separate file and is not part of Stage 4A")
            print("NOT runnable: expert carrier, layer math and Runtime still separate")
            return 0
        chunk = args.chunk_mib << 20
        if args.verify:
            result = verify_remaining(checkpoint, args.output,
                                      includes=tuple(args.include),
                                      require_complete=args.require_complete,
                                      chunk_bytes=chunk)
            print(f"VERIFY OK: {result['tensors']} main-checkpoint non-expert tensors; "
                  "exact byte comparison to HF source")
            return 0
        max_file_bytes = None if args.max_file_mib is None else args.max_file_mib << 20
        result = pack_remaining(checkpoint, args.output,
                                includes=tuple(args.include),
                                require_complete=args.require_complete,
                                chunk_bytes=chunk, max_file_bytes=max_file_bytes)
        print(f"PACK OK: {result['tensors']} non-expert tensors; "
              f"{result['payload_bytes'] / (1 << 30):.3f} GiB payload")
        print(f"NInfer v3 NON-EXECUTABLE remainder carrier: {result['output']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

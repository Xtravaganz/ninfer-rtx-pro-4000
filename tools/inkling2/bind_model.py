#!/usr/bin/env python3
"""Inkling 2 Mini Stage4B: validate two weight carriers and write a non-executable role plan."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support direct `python3 tools/inkling2/bind_model.py`, without PYTHONPATH.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonexpert_artifact import Checkpoint
from runtime_bindings import create_binding_plan, write_plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="original complete HF snapshot directory")
    p.add_argument("--nonexpert", type=Path, help="Stage-4A NInfer-v3 carrier, partial or complete")
    p.add_argument("--experts", type=Path, help="Stage-2B NInfer-v3 carrier, partial or complete")
    p.add_argument("--banks", type=Path, help="matching Stage-2A directory (binds provenance)")
    p.add_argument("--output", type=Path, help="write JSON plan; never overwrite")
    p.add_argument("--plan", action="store_true", help="HF metadata-only role/schedule plan, no carriers")
    p.add_argument("--require-complete", action="store_true", help="fail if a carrier/role/expert is missing")
    p.add_argument("--verify-bytes", action="store_true", help="bounded source-to-v3 byte comparisons")
    args = p.parse_args(argv)
    if args.plan and any((args.nonexpert, args.experts, args.banks, args.verify_bytes)):
        p.error("--plan cannot use carrier options")
    if args.banks and not args.experts:
        p.error("--banks requires --experts")
    if args.verify_bytes and not (args.nonexpert and args.experts and args.banks):
        p.error("--verify-bytes requires --nonexpert, --experts and --banks")

    with Checkpoint(args.source) as checkpoint:
        if args.plan:
            report = create_binding_plan(checkpoint)
        else:
            from contextlib import ExitStack
            from tools.artifact.reader import Artifact
            with ExitStack() as stack:
                remainder = stack.enter_context(Artifact(args.nonexpert)) if args.nonexpert else None
                experts = stack.enter_context(Artifact(args.experts)) if args.experts else None
                report = create_binding_plan(checkpoint, nonexpert=remainder, experts=experts,
                                             banks_dir=args.banks)
            if args.verify_bytes:
                from nonexpert_artifact import verify_remaining
                from ninfer_artifact_bridge import verify_artifact, load_manifest
                from native_experts import IndexedSafetensors, describe_expert_bank, verify_bank
                verify_remaining(checkpoint, args.nonexpert, require_complete=False)
                verify_artifact(args.banks, args.experts)
                with IndexedSafetensors(args.source) as store:
                    for bank in load_manifest(args.banks):
                        original = describe_expert_bank(store, bank.layer, bank.projection)
                        verify_bank(store, original, bank.path, expected_sha256=bank.digest)
                report["coverage"]["weight_bytes_verified"] = True
                report["coverage"]["verification_scope"] = (
                    "All selected Stage4A bytes vs HF; Stage2B vs Stage2A bank files; "
                    "Stage2A selected expert bytes vs original HF source."
                )
        cov = report["coverage"]
        print(f"HF: {report['checkpoint']['tensors']} indexed tensors; "
              f"nonexpert={cov['nonexpert_expected']}; expected expert banks=48")
        print(f"PLAN: {len(report['roles'])} nonexpert runtime roles; "
              "48 logical visits and 48 cache slots")
        print(f"BIND: nonexpert={cov['nonexpert_bound']}/{cov['nonexpert_expected']} "
              f"expert_banks={cov['expert_banks_bound']}/48 "
              f"stage2a_manifest_match={cov['bank_manifest_matches']} "
              f"unresolved_roles={len(cov['unresolved_roles'])}")
        if cov["unresolved_roles"]:
            print("UNRESOLVED (first 12): " + ", ".join(cov["unresolved_roles"][:12]))
        print("COVERAGE: " + ("COMPLETE METADATA" if cov["metadata_complete"] else "PARTIAL / NOT FULLY CLASSIFIED"))
        print("RUNTIME: NOT LOADABLE (Inkling execution operators not yet integrated)")
        if args.require_complete and not cov["metadata_complete"]:
            raise ValueError("--require-complete: model binding coverage is incomplete")
        if args.output:
            path = write_plan(args.output, report)
            print(f"WROTE: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

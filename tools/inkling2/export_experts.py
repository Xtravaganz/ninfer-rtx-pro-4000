#!/usr/bin/env python3
"""Export Inkling 2 Mini expert banks without materializing/requantizing weights.

Examples:
  python3 tools/inkling2/export_experts.py --plan --config /tmp/inkling2-config.json --index /tmp/inkling2-index.json
  python3 tools/inkling2/export_experts.py --source /data/Inkling-2-Mini --output /data/inkling2-native --layers 0,20 --experts 0-1
  python3 tools/inkling2/export_experts.py --source /data/Inkling-2-Mini --output /data/inkling2-native --verify
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from native_experts import (
    ExpertTensors,
    FMT_NVFP4,
    IndexedSafetensors,
    MiniGeometry,
    describe_expert_bank,
    export_bank,
    format_for_layer,
    parse_selection,
    verify_bank,
)
from check_checkpoint import validate_config, validate_header, validate_index


def file_name(layer: int, projection: str) -> str:
    return f"layer{layer:02d}_{projection}.i2bank"


def expected_bank_size(layer: int, projection: str, expert_count: int,
                       geometry: MiniGeometry = MiniGeometry()) -> int:
    k, n = geometry.cols(projection), geometry.rows
    if format_for_layer(layer) == FMT_NVFP4:
        code = n * k // 2
        scale = n * k // 16
        aux = 8
    else:
        code = n * k
        scale = n * k // 32
        aux = 0
    return 64 + expert_count * 4 + expert_count * (code + scale + aux)


def check_header_subset(header: dict, layers: tuple[int,...], geom: MiniGeometry) -> None:
    """Check actual model header without opening 19.5GB of weights."""
    for layer in layers:
        for projection in ("w13", "w2"):
            k = geom.cols(projection)
            stem = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
            fmt = format_for_layer(layer)
            for name, expected in (
                (stem, ("U8" if fmt == FMT_NVFP4 else "F8_E4M3", [geom.experts, geom.rows, k//2 if fmt == FMT_NVFP4 else k])),
                (stem + "_scale", ("F8_E4M3" if fmt == FMT_NVFP4 else "U8", [geom.experts, geom.rows, k//(16 if fmt == FMT_NVFP4 else 32)])),
            ):
                if name not in header or (header[name].get("dtype"), header[name].get("shape")) != expected:
                    raise ValueError(f"{name}: incompatible dtype/shape in supplied header")
            if fmt == FMT_NVFP4:
                for suffix, expected in (("_amax", ("F32", [geom.experts])),
                                         ("_outer_scale_denominator", ("F32", [1]))):
                    item = header.get(stem+suffix, {})
                    if (item.get("dtype"), item.get("shape")) != expected:
                        raise ValueError(f"{stem+suffix}: incompatible dtype/shape")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, help="Local official model directory containing safetensors and index")
    p.add_argument("--config", type=Path, help="Original HF config.json (defaults to --source/config.json)")
    p.add_argument("--index", type=Path, help="HF model.safetensors.index.json (defaults to --source/...)")
    p.add_argument("--header", type=Path, help="Optional extracted safetensors header JSON for plan-time dtype/shape checks")
    p.add_argument("--output", type=Path, help="Destination outside the HF checkpoint and Git repository")
    p.add_argument("--layers", default="all", help="all, individual indices or ranges, e.g. 0,6,20-23")
    p.add_argument("--experts", default="all", help="all, individual indices or ranges, e.g. 0-3,7")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="No weight reads, no writes: estimate output and validate index")
    action.add_argument("--verify", action="store_true", help="Byte-compare existing output banks against the source")
    p.add_argument("--force", action="store_true", help="Explicitly replace any existing bank files/manifest")
    p.add_argument("--chunk-mib", type=int, default=4, help="Maximum source read allocation in MiB (default 4)")
    a = p.parse_args(argv)
    root = a.source
    config_path = a.config or (root / "config.json" if root else None)
    index_path = a.index or (root / "model.safetensors.index.json" if root else None)
    if config_path is None or index_path is None:
        p.error("--config and --index are required without --source")
    if not a.plan and (root is None or a.output is None):
        p.error("export/verify require --source and --output")
    if not a.plan and a.header:
        p.error("--header is only for --plan; file validation uses actual source shard")
    if not 1 <= a.chunk_mib <= 1024:
        p.error("--chunk-mib must be between 1 and 1024")
    config, index = json.loads(config_path.read_text()), json.loads(index_path.read_text())
    validate_config(config)
    validate_index(index)
    geom = MiniGeometry()
    layers = parse_selection(a.layers, 24)
    experts = parse_selection(a.experts, geom.experts)
    if a.header:
        check_header_subset(json.loads(a.header.read_text()), layers, geom)
        print("Header dtype/shape: OK")
    total_bytes = sum(expected_bank_size(layer, projection, len(experts), geom)
                      for layer in layers for projection in ("w13", "w2"))
    print(f"Inkling 2 Mini: {len(layers)} layers × 2 projections × {len(experts)} experts; "
          f"estimated output {total_bytes / (1024**3):.3f} GiB")
    if a.plan:
        print("PLAN ONLY: checkpoint names checked; no weight bytes read or written")
        if not a.header:
            print("NOTE: dtype/shape not checked; supply --header for offline metadata validation")
        return 0

    assert root is not None and a.output is not None
    output = a.output.resolve()
    source = root.resolve()
    if output == source or source in output.parents:
        p.error("--output must be outside --source (large checkpoint directory)")
    if a.verify:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("version") != 1 or manifest.get("layout") != "inkling2_native_expert_bank_v1":
            raise ValueError("unsupported/missing native expert bank manifest")
        with IndexedSafetensors(source) as store:
            count = 0
            for entry in manifest.get("banks", []):
                if entry["layer"] not in layers:
                    continue
                if not set(entry["expert_ids"]).issubset(experts):
                    continue
                path = (output / entry["file"]).resolve()
                if path.parent != output:
                    raise ValueError("unsafe manifest file path")
                bank = describe_expert_bank(store, entry["layer"], entry["projection"])
                verify_bank(store, bank, path, entry["sha256"], a.chunk_mib * 1024 * 1024)
                print(f"VERIFY OK {entry['file']} {len(entry['expert_ids'])} experts")
                count += 1
        if count == 0:
            raise ValueError("no manifest banks matched selection")
        print(f"OK: {count} banks byte-identical to source")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not a.force:
        raise FileExistsError(f"{manifest_path}: existing output; use --force to replace")
    with IndexedSafetensors(source) as store:
        # Preflight all requested tensor names/dtypes/shapes before writing any output.
        banks = [describe_expert_bank(store, layer, proj, geom)
                 for layer in layers for proj in ("w13", "w2")]
        existing = [file_name(bank.layer, bank.projection) for bank in banks
                    if (output / file_name(bank.layer, bank.projection)).exists()]
        if existing and not a.force:
            raise FileExistsError(f"output banks already exist: {existing[:4]} (use --force)")
        entries = []
        for bank in banks:
            dest = output / file_name(bank.layer, bank.projection)
            entry = export_bank(store, bank, experts, dest, a.chunk_mib * 1024 * 1024, a.force)
            entries.append(entry)
            print(f"EXPORT OK {entry['file']}: {entry['bytes'] / (1024**2):.1f} MiB "
                  f"SHA256 {entry['sha256'][:16]}", flush=True)
    manifest = {
        "version": 1,
        "layout": "inkling2_native_expert_bank_v1",
        "format_note": "Staging format: original bits, not a loadable NInfer artifact; NVFP4 scales unswizzled",
        "source_architecture": "Inkling2LoopForConditionalGeneration",
        "source_config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "source_index_sha256": sha256(index_path.read_bytes()).hexdigest(),
        "geometry": {"experts": geom.experts, "rows": geom.rows, "w13_cols": geom.gate_up_cols, "w2_cols": geom.down_cols},
        "banks": entries,
    }
    data = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with __import__("tempfile").NamedTemporaryFile(mode="w", dir=output, prefix="manifest.",
                                                 suffix=".part", delete=False) as f:
        temp = Path(f.name)
        f.write(data)
        f.flush()
        __import__("os").fsync(f.fileno())
    temp.replace(manifest_path)
    print(f"DONE: wrote {len(entries)} banks; manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

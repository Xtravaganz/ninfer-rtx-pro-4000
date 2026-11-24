#!/usr/bin/env python3
"""Select a few real Inkling experts from both verified NInfer-v3 carriers.

Output is an *isolated experiment bundle*, not a runnable ninfer model.
Produces no generated tokens and does not change NInfer artifacts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_carrier_stage5f import (
    Carrier5F, HIDDEN, INTERMEDIATE, KIND_NVFP4, KIND_MXFP8,
    TOTAL_EXPERTS, TOP_K, f32_bytes, fp32_values, router_scores,
)

MAGIC = b"I2REALF1"
VERSION = 1
HEADER = struct.Struct("<8s7I")
FILE_SCHEME = {
    "header.bin": None,
    "hidden.f32": HIDDEN * 4,
    "router.f32": TOTAL_EXPERTS * HIDDEN * 4,
    "bias.f32": TOTAL_EXPERTS * 4,
    "resident_map.i32": TOTAL_EXPERTS * 4,
    "predicted_ids.i32": TOP_K * 4,
    "predicted_weights.f32": TOP_K * 4,
    "route_scale.f32": 4,
}


def expected_sizes(fmt: int, resident: int) -> dict[str, int]:
    if fmt not in (KIND_NVFP4, KIND_MXFP8) or not 8 <= resident <= 128:
        raise ValueError("invalid bundle precision or resident count")
    result = dict(FILE_SCHEME)
    result["header.bin"] = HEADER.size
    for prefix, k in (("w13", HIDDEN), ("w2", INTERMEDIATE)):
        result[f"{prefix}_codes.u8"] = resident * HIDDEN * k // (2 if fmt == KIND_NVFP4 else 1)
        result[f"{prefix}_scales.u8"] = resident * HIDDEN * k // (16 if fmt == KIND_NVFP4 else 32)
        if fmt == KIND_NVFP4:
            result[f"{prefix}_amax.f32"] = resident * 4
            result[f"{prefix}_denominator.f32"] = resident * 4
    return result


def verify_bundle(root: Path) -> dict:
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_bytes())
    if meta.get("contract") != "inkling2_real_moe_stage5fg_v1":
        raise ValueError("unsupported Stage-5F bundle manifest")
    hdr = (root / "header.bin").read_bytes()
    if len(hdr) != HEADER.size:
        raise ValueError("invalid bundle header size")
    magic, version, layer, fmt, count, n, k, token = HEADER.unpack(hdr)
    if (magic != MAGIC or version != VERSION or not 0 <= layer < 24 or
        fmt != (KIND_NVFP4 if layer < 20 else KIND_MXFP8) or
        n != HIDDEN or k != INTERMEDIATE or not 0 <= token < 201024):
        raise ValueError("inconsistent real bundle header")
    sizes = expected_sizes(fmt, count)
    if set(meta.get("files", {})) != set(sizes):
        raise ValueError("missing or additional bundle files")
    for filename, size in sizes.items():
        desc = meta["files"][filename]
        if desc.get("bytes") != size or (root / filename).stat().st_size != size:
            raise ValueError(f"invalid size for {filename}")
        h = sha256()
        with (root / filename).open("rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                h.update(chunk)
        if h.hexdigest() != desc.get("sha256"):
            raise ValueError(f"checksum mismatch for {filename}")
    resident_map = struct.unpack("<128i", (root / "resident_map.i32").read_bytes())
    entries = sorted((index, pos) for index, pos in enumerate(resident_map) if pos >= 0)
    if len(entries) != count or sorted(pos for _, pos in entries) != list(range(count)):
        raise ValueError("invalid resident map")
    if meta.get("resident_ids") != [eid for eid, _ in sorted(entries, key=lambda p: p[1])]:
        raise ValueError("resident IDs differ from map")
    for eid in struct.unpack("<8i", (root / "predicted_ids.i32").read_bytes()):
        if eid < 0 or resident_map[eid] < 0:
            raise ValueError("predicted router ID is not resident")
    return meta


def create_bundle(*, checkpoint_root: Path, nonexpert_path: Path,
                  experts_path: Path, banks_path: Path,
                  output_path: Path, layer: int, token_id: int,
                  count: int, hidden_scale: float = 1.0) -> dict:
    if type(layer) is not int or not 0 <= layer < 24:
        raise ValueError("layer must be physical layer 0..23")
    if type(token_id) is not int or not 0 <= token_id < 201024:
        raise ValueError("token id out of range")
    fmt = KIND_NVFP4 if layer < 20 else KIND_MXFP8
    expected_sizes(fmt, count)
    if not math.isfinite(hidden_scale) or not 0 < hidden_scale <= 10:
        raise ValueError("input hidden scale must be finite, positive, <=10")
    output_path = Path(output_path).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    from tools.artifact.reader import Artifact
    from real_carrier_stage5f import validate_complete_carriers
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".stage5fg-", dir=output_path.parent))
    published = False
    try:
        with Artifact(nonexpert_path) as nonexp, Artifact(experts_path) as exps:
            plan = validate_complete_carriers(checkpoint_root, nonexp, exps, banks_path)
            reader = Carrier5F(nonexp, exps)
            hidden = tuple(hidden_scale * x for x in reader.embedding_row(token_id))
            router = reader.nonexpert_tensor_f32(
                f"model.llm.layers.{layer}.mlp.gate.weight", (TOTAL_EXPERTS, HIDDEN))
            bias = reader.nonexpert_tensor_f32(
                f"model.llm.layers.{layer}.mlp.gate.bias", (TOTAL_EXPERTS,))
            scores = router_scores(hidden, router, bias)
            # Stage-5E resident map has a CPU/GPU float-rounding guard band:
            # stage more than top-8, never allow nonresident GPU ids silently.
            selected = [e for _, _, e in scores[:TOP_K]]
            resident = sorted(e for _, _, e in scores[:count])
            rank_to_index = {e: i for i, e in enumerate(resident)}
            mapping = [rank_to_index.get(e, -1) for e in range(TOTAL_EXPERTS)]
            route_scale = float(json.loads((Path(checkpoint_root) / "config.json").read_text())["text_config"]["route_scale"])
            if route_scale <= 0 or not math.isfinite(route_scale):
                raise ValueError("invalid route_scale")
            total_score = math.fsum(score for _, score, _ in scores[:TOP_K])
            if total_score <= 0:
                raise ValueError("zero router score mass")
            weights = [score * route_scale / (total_score + 1e-20)
                       for _, score, _ in scores[:TOP_K]]

            def write(name: str, data: bytes) -> None:
                (temp / name).write_bytes(data)

            write("header.bin", HEADER.pack(MAGIC, VERSION, layer, fmt, count,
                                             HIDDEN, INTERMEDIATE, token_id))
            write("hidden.f32", f32_bytes(hidden))
            write("router.f32", router)
            write("bias.f32", bias)
            write("resident_map.i32", struct.pack("<128i", *mapping))
            write("predicted_ids.i32", struct.pack("<8i", *selected))
            write("predicted_weights.f32", f32_bytes(weights))
            write("route_scale.f32", struct.pack("<f", route_scale))
            for projection in ("w13", "w2"):
                with (temp / f"{projection}_codes.u8").open("wb") as codes, \
                     (temp / f"{projection}_scales.u8").open("wb") as scales:
                    amax_values, denominator_values = [], []
                    for expert_id in resident:
                        part = reader.bank(layer, projection, [expert_id])[0]
                        codes.write(part.codes)
                        scales.write(part.scales)
                        if fmt == KIND_NVFP4:
                            amax_values.append(part.amax)
                            denominator_values.append(part.denominator)
                if fmt == KIND_NVFP4:
                    write(f"{projection}_amax.f32", f32_bytes(amax_values))
                    write(f"{projection}_denominator.f32", f32_bytes(denominator_values))
            sizes = expected_sizes(fmt, count)
            files = {}
            for name, expected in sizes.items():
                target = temp / name
                actual = target.stat().st_size
                if actual != expected:
                    raise ValueError(f"{name}: {actual} bytes instead of {expected}")
                digest = sha256()
                with target.open("rb") as f:
                    while chunk := f.read(4 * 1024 * 1024):
                        digest.update(chunk)
                files[name] = {"bytes": actual, "sha256": digest.hexdigest()}
            meta = {
                "contract": "inkling2_real_moe_stage5fg_v1",
                "runtime_loadable": False,
                "description": "One MoE operator on real weights, NOT full model inference",
                "layer": layer,
                "token_id": token_id,
                "hidden_input": "scaled BF16 embedding row; not a decoder hidden state",
                "hidden_scale": hidden_scale,
                "format": "nvfp4_e2m1_g16" if fmt == KIND_NVFP4 else "mxfp8_e4m3_g32",
                "resident_ids": resident,
                "predicted_top8_ids": selected,
                "checkpoint_index_sha256": plan["checkpoint"]["model_index_sha256"],
                "nonexpert_artifact_id": nonexp.artifact_id.hex(),
                "experts_artifact_id": exps.artifact_id.hex(),
                "stage2a_manifest_matches": plan["coverage"]["bank_manifest_matches"],
                "files": files,
            }
            (temp / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        verify_bundle(temp)
        os.rename(temp, output_path)
        published = True
        return meta
    finally:
        if not published:
            shutil.rmtree(temp, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, help="complete original HF snapshot")
    p.add_argument("--nonexpert", type=Path, help="Stage-4A complete nonexpert .ninfer")
    p.add_argument("--experts", type=Path, help="Stage-2B complete experts .ninfer")
    p.add_argument("--banks", type=Path, help="matching Stage-2A full bank directory")
    p.add_argument("--layer", type=int, default=0, help="physical layer; test 0 and 20")
    p.add_argument("--token-id", type=int, default=1000)
    p.add_argument("--resident", type=int, default=32, help="top N CPU-scored experts to stage (8..128)")
    p.add_argument("--hidden-scale", type=float, default=1.0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--verify", action="store_true", help="verify bundle hashes and geometry; no carriers needed")
    args = p.parse_args(argv)
    if args.verify:
        meta = verify_bundle(args.output)
        print(f"VERIFY OK Stage5F: layer={meta['layer']} resident={len(meta['resident_ids'])} real expert pairs; no generation")
        return 0
    if not all((args.source, args.nonexpert, args.experts, args.banks)):
        p.error("creation requires --source --nonexpert --experts --banks")
    meta = create_bundle(checkpoint_root=args.source, nonexpert_path=args.nonexpert,
                         experts_path=args.experts, banks_path=args.banks,
                         output_path=args.output, layer=args.layer, token_id=args.token_id,
                         count=args.resident, hidden_scale=args.hidden_scale)
    print(f"PACK OK Stage5F: layer={meta['layer']} format={meta['format']} "
          f"resident={len(meta['resident_ids'])}/128 real experts; token={meta['token_id']}")
    print(f"BUNDLE: {args.output} | NOT FULL MODEL INFERENCE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Inkling 2 Mini Stage 4B: fail-closed metadata bindings across two NInfer v3 carriers.

This is a *binding plan*, NOT a model format or executable runtime registry.
No weight payload bytes are required for planning/binding; optional --verify-bytes
runs the existing Stage-2B/4A source-vs-artifact verifiers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from typing import Any

from nonexpert_artifact import Checkpoint, STORAGE_CONTRACT, is_expert_source
from ninfer_artifact_bridge import LAYOUT as EXPERT_CONTRACT

CONTRACT = "inkling2_runtime_binding_plan_v1"
LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
QUERY = re.compile(r"^model\.llm\.attn_res\.(attn_queries|mlp_queries)\.(\d+)$")
# Strict HF-original loop controllers: 12 core layers x 4 streams x 3 matrices.
CONTROLLER = re.compile(r"^controllers\.(attn|k|v|mlp)\.(down|forget|gain)_weight$")
# Three independent input normalizers on the four full-attention layers.
FULL_INPUT_NORMS = frozenset(("q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"))
HIDDEN = 2304
LOOP_GATE_RANK = 64
BASE = {
    "model.llm.embed.weight": "embedding",
    "model.llm.embed_norm.weight": "embedding_norm",
    "model.llm.unembed.weight": "lm_head",
    "model.llm.norm.weight": "final_norm",
}
GLOBAL = {
    "model.llm.attn_res.output_query": "output_residual",
    "model.llm.middle_cycle_cross_pass_queries": "cross_pass_query",
    "model.llm.middle_cycle_cross_pass_kv_queries": "cross_pass_kv_query",
}


@dataclass(frozen=True, slots=True)
class Role:
    source: str
    role: str
    category: str
    layer: int | None
    dtype: str
    shape: tuple[int, ...]

    def json(self) -> dict:
        return {**asdict(self), "shape": list(self.shape)}


def classify(name: str, dtype: str, shape: tuple[int, ...]) -> Role:
    """Classify without renaming/reshaping any checkpoint weights.

    Unknown names remain explicit unresolved roles. This prevents a future
    checkpoint revision being silently treated as a supported operator.
    """
    if is_expert_source(name):
        raise ValueError(f"{name}: packed expert source must NOT be in non-expert roles")
    if name in BASE:
        return Role(name, f"text.{BASE[name]}", BASE[name], None, dtype, shape)
    if name in GLOBAL:
        return Role(name, f"text.{GLOBAL[name]}", "loop_control" if "cross_pass" in name else "block_residual", None, dtype, shape)
    query = QUERY.fullmatch(name)
    if query:
        kind, i = query.groups()
        pos = int(i)
        if pos >= 24:
            raise ValueError(f"{name}: residual query index outside 24 physical layers")
        return Role(name, f"text.block_residual.{kind}.{pos}", "block_residual", pos, dtype, shape)
    layer = LAYER.fullmatch(name)
    if layer:
        layer_id, tail = int(layer.group(1)), layer.group(2)
        if not 0 <= layer_id < 24:
            raise ValueError(f"{name}: physical layer index outside [0,24)")
        controller = CONTROLLER.fullmatch(tail)
        if controller is not None:
            stream, part = controller.groups()
            if not 6 <= layer_id < 18:
                raise ValueError(f"{name}: loop controller outside physical core [6,18)")
            expected = ((LOOP_GATE_RANK, 2 * HIDDEN) if part == "down"
                        else (HIDDEN, LOOP_GATE_RANK))
            if dtype != "BF16" or shape != expected:
                raise ValueError(
                    f"{name}: expected BF16 {expected} for {stream}/{part}, "
                    f"got {dtype} {shape}"
                )
            category = "loop_control"
        elif tail in FULL_INPUT_NORMS:
            if layer_id not in (5, 11, 17, 23):
                raise ValueError(f"{name}: full-attention input norm on sliding layer")
            if dtype not in ("F32", "BF16") or shape != (HIDDEN,):
                raise ValueError(
                    f"{name}: expected FP32/BF16 ({HIDDEN},), got {dtype} {shape}"
                )
            category = "normalization"
        elif tail.startswith("mlp.gate."):
            category = "router"
        elif "message_gate" in tail or "cross_pass" in tail:
            category = "loop_control" if 6 <= layer_id < 18 else "unresolved"
        elif tail.startswith("attn."):
            category = "attention"
        elif tail.startswith(("attn_sconv.", "mlp_sconv.", "post_attention_convolution.", "post_mlp_convolution.")):
            category = "short_convolution"
        elif tail.startswith(("attn_norm.", "mlp_norm.", "input_layernorm.", "post_attention_layernorm.",
                              "q_input_layernorm.", "k_input_layernorm.", "v_input_layernorm.")):
            category = "normalization"
        elif tail.startswith(("attn_res.", "attention_input.", "mlp_input.")):
            category = "block_residual"
        elif tail.startswith("mlp."):
            category = "mlp_auxiliary"
        else:
            category = "unresolved"
        return Role(name, f"text.layer{layer_id:02d}.{tail}", category, layer_id, dtype, shape)
    if name.startswith(("model.visual.", "model.vision.")):
        return Role(name, "vision." + name.split(".", 2)[2], "vision", None, dtype, shape)
    return Role(name, "unresolved." + name, "unresolved", None, dtype, shape)


def classify_all(checkpoint: Checkpoint) -> list[Role]:
    roles = [classify(span.name, span.dtype, span.shape)
             for span in (checkpoint.spans[n] for n in checkpoint.remaining_names)]
    if len({r.role for r in roles}) != len(roles):
        raise ValueError("different HF tensors resolve to the same runtime role")
    return roles


def schedule(config: dict) -> list[dict]:
    """Python independent reproduction of Stage-1 C++ 48-step/cache contract."""
    t = config["text_config"]
    loop = t["loop_config"]
    if (t["num_hidden_layers"] != 24 or loop !=
        {"core_start": 6, "core_end": 18, "num_passes": 3,
         "gate_rank": 64, "is_varloop": False} or
        t["n_layers_per_attn_res_block"] != 6 or
        t["local_layer_ids"] != [i for i in range(24) if i % 6 != 5]):
        raise ValueError("unsupported Inkling 2 Mini execution schedule")
    steps = []
    def append(stage: str, layer: int, pass_id: int, slot: int):
        steps.append({
            "ordinal": len(steps), "stage": stage, "physical_layer": layer,
            "core_pass": pass_id, "cache_slot": slot,
            "attention": "sliding" if layer in t["local_layer_ids"] else "full",
            "block_boundary": layer % 6 == 0,
            "reads_previous_pass_messages": stage == "core" and pass_id > 0,
            "reset_to_prelude": stage == "core" and pass_id > 0 and layer == 6,
            "layer_role_prefix": f"text.layer{layer:02d}.",
            "expert_bank_ids": [f"inkling2/layer{layer:02d}/w13", f"inkling2/layer{layer:02d}/w2"],
        })
    for i in range(6):
        append("prelude", i, 0, i)
    for p in range(3):
        for i in range(6, 18):
            append("core", i, p, i if p == 0 else 24 + (p - 1) * 12 + i - 6)
    for i in range(18, 24):
        append("postlude", i, 0, i)
    if len(steps) != 48 or sorted(s["cache_slot"] for s in steps) != list(range(48)):
        raise AssertionError("Inkling2 loop schedule inconsistent")
    return steps


def _check_config(artifact: Any, checkpoint: Checkpoint, label: str) -> None:
    component = artifact.directory.components.get("text", {})
    actual = component.get("config", {})
    raw = checkpoint.config["text_config"]
    if (not isinstance(actual, dict) or actual.get("model_type") != "inkling2_loop_text" or
        actual.get("architectures") != checkpoint.config["architectures"] or
        actual.get("eos_token_id") != checkpoint.config["eos_token_id"] or
        any(actual.get(k) != v for k, v in raw.items())):
        raise ValueError(f"{label}: model config does not match HF checkpoint")
    if artifact.directory.metadata.get("runtime_loadable") is not False:
        raise ValueError(f"{label}: expected non-executable weight carrier")


def inspect_nonexpert(artifact: Any, checkpoint: Checkpoint) -> tuple[list[dict], bool]:
    """Validate every listed tensor descriptor against original HF index + header."""
    meta = artifact.directory.metadata
    if meta.get("storage_contract") != STORAGE_CONTRACT or meta.get("main_checkpoint_only") is not True:
        raise ValueError("not an Inkling Stage-4A non-expert carrier")
    if artifact.directory.provenance.get("model_index_sha256") != checkpoint.index_sha256:
        raise ValueError("Stage-4A artifact belongs to a different HF index")
    _check_config(artifact, checkpoint, "nonexpert")
    records = meta.get("tensors")
    if not isinstance(records, list) or not records:
        raise ValueError("Stage-4A has no tensor records")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("malformed Stage-4A tensor record")
        name = record.get("name")
        if not isinstance(name, str) or name in names or name not in checkpoint.remaining_names:
            raise ValueError(f"duplicate or unexpected Stage-4A tensor: {name!r}")
        names.add(name)
        span = checkpoint.spans[name]
        if record != span.record():
            raise ValueError(f"Stage-4A descriptor does not match HF: {name}")
        obj = artifact.object(span.obj_id)
        if obj.bytes != span.byte_count:
            raise ValueError(f"Stage-4A object byte count mismatch: {name}")
        if span.direct_format:
            if (obj.kind != "tensor" or obj.shape != span.shape or obj.format != span.direct_format or
                obj.layout != "contiguous_le_v1" or
                artifact.directory.bindings.get(name) != {"object": span.obj_id}):
                raise ValueError(f"Stage-4A native binding mismatch: {name}")
        elif (obj.kind != "resource" or obj.encoding != "raw_bytes_v1" or
              artifact.directory.components["text"].get("resources", {}).get(span.obj_id) != span.obj_id):
            raise ValueError(f"Stage-4A raw resource mismatch: {name}")
    if len(artifact.objects) != len(records) or set(artifact.by_id) != {"hf/" + n for n in names}:
        raise ValueError("Stage-4A has extra or missing physical objects")
    if set(artifact.directory.bindings) != {n for n in names if checkpoint.spans[n].direct_format}:
        raise ValueError("Stage-4A contains unexpected bindings")
    return [{"role": classify(checkpoint.spans[r["name"]].name,
                                checkpoint.spans[r["name"]].dtype,
                                checkpoint.spans[r["name"]].shape).role,
             "source": r["name"], "object": r["object"], "carrier": "nonexpert"}
            for r in records], names == set(checkpoint.remaining_names)


def _check_object(artifact: Any, name: str, *, kind: str, num_bytes: int | None = None,
                  shape: tuple[int, ...] | None = None, fmt: str | None = None,
                  layout: str | None = None) -> None:
    obj = artifact.object(name)
    if obj.kind != kind or (num_bytes is not None and obj.bytes != num_bytes):
        raise ValueError(f"expert bank object invalid: {name}")
    if kind == "tensor" and (obj.shape != shape or obj.format != fmt or obj.layout != layout):
        raise ValueError(f"expert bank tensor layout invalid: {name}")
    if kind == "resource" and obj.encoding != "raw_bytes_v1":
        raise ValueError(f"expert bank resource encoding invalid: {name}")


def inspect_experts(artifact: Any, checkpoint: Checkpoint, *, banks_dir: Path | None = None,
                    read_id_bytes: bool = True) -> tuple[list[dict], bool, bool]:
    """Validate Stage-2B records without reading the large quantized planes."""
    meta = artifact.directory.metadata
    if meta.get("storage_contract") != EXPERT_CONTRACT:
        raise ValueError("not an Inkling Stage-2B expert carrier")
    _check_config(artifact, checkpoint, "expert")
    records = meta.get("inkling2_banks")
    if not isinstance(records, list) or not records:
        raise ValueError("Stage-2B has no bank records")
    expected_ids = set()
    seen = set()
    full = True
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("invalid Stage-2B record")
        layer, proj = rec.get("layer"), rec.get("projection")
        if type(layer) is not int or not 0 <= layer < 24 or proj not in ("w13", "w2"):
            raise ValueError("invalid Stage-2B layer/projection")
        key = (layer, proj)
        if key in seen:
            raise ValueError(f"duplicate Stage-2B bank: {key}")
        seen.add(key)
        ids = rec.get("expert_ids")
        if (not isinstance(ids, list) or not ids or
            any(type(i) is not int or not 0 <= i < 128 for i in ids) or
            ids != sorted(set(ids))):
            raise ValueError(f"invalid expert ID selection at {key}")
        full &= ids == list(range(128))
        n, k = 2304, 2304 if proj == "w13" else 1152
        prefix = f"inkling2/layer{layer:02d}/{proj}"
        if rec.get("shape") != [len(ids), n, k]:
            raise ValueError(f"Stage-2B bank geometry mismatch: {key}")
        id_name = prefix + "/expert_ids"
        _check_object(artifact, id_name, kind="resource", num_bytes=len(ids) * 4)
        expected_ids.add(id_name)
        if read_id_bytes and artifact.read_object(id_name) != struct.pack(f"<{len(ids)}I", *ids):
            raise ValueError(f"Stage-2B expert ID bytes disagree with bank descriptor: {key}")
        if layer < 20:
            if rec.get("quantization") != "nvfp4_e2m1_g16_e4m3":
                raise ValueError(f"NVFP4 bank format mismatch: {key}")
            tensor = prefix + "/nvfp4"
            scalars = prefix + "/original_amax_denominator_fp32"
            if rec.get("native_tensor") != tensor or rec.get("original_scalars") != scalars:
                raise ValueError(f"NVFP4 object mapping mismatch: {key}")
            _check_object(artifact, tensor, kind="tensor", shape=(len(ids), n, k), fmt="nvfp4",
                          layout="batched_block_scale_k16_m128x4_v1")
            _check_object(artifact, scalars, kind="resource", num_bytes=len(ids) * 8)
            expected_ids.update((tensor, scalars))
            if artifact.directory.bindings.get(tensor) != {"object": tensor}:
                raise ValueError(f"NVFP4 missing native v3 binding: {key}")
        else:
            if rec.get("quantization") != "mxfp8_e4m3_g32_ue8m0" or rec.get("scale_group") != 32 or rec.get("scale_dtype") != "UE8M0_U8":
                raise ValueError(f"MXFP8 bank format mismatch: {key}")
            codes, scales = prefix + "/mxfp8_codes", prefix + "/mxfp8_ue8m0_scales"
            if rec.get("codes") != codes or rec.get("scales") != scales:
                raise ValueError(f"MXFP8 object mapping mismatch: {key}")
            _check_object(artifact, codes, kind="resource", num_bytes=len(ids) * n * k)
            _check_object(artifact, scales, kind="resource", num_bytes=len(ids) * n * (k // 32))
            expected_ids.update((codes, scales))
    if set(artifact.by_id) != expected_ids or len(artifact.objects) != len(expected_ids):
        raise ValueError("Stage-2B has extra/missing bank objects")
    resources = artifact.directory.components["text"].get("resources", {})
    expected_resources = {id for id in expected_ids if artifact.object(id).kind == "resource"}
    if set(resources) != expected_resources or any(resources.get(id) != id for id in expected_resources):
        raise ValueError("Stage-2B resource map is inconsistent with bank objects")
    native_bindings = {f"inkling2/layer{i:02d}/{p}/nvfp4"
                       for i, p in seen if i < 20}
    if set(artifact.directory.bindings) != native_bindings:
        raise ValueError("Stage-2B native binding set does not match NVFP4 banks")
    full &= len(seen) == 48
    origin_confirmed = False
    if banks_dir is not None:
        manifest = (Path(banks_dir) / "manifest.json").read_bytes()
        metadata = json.loads(manifest)
        if (metadata.get("source_index_sha256") != checkpoint.index_sha256 or
            metadata.get("source_config_sha256") != sha256((checkpoint.root / "config.json").read_bytes()).hexdigest()):
            raise ValueError("Stage-2A manifest was exported from a different HF snapshot")
        if artifact.directory.provenance.get("stage2a_manifest_sha256") != sha256(manifest).hexdigest():
            raise ValueError("Stage-2B does not belong to this Stage-2A bank directory")
        from ninfer_artifact_bridge import load_manifest
        banks = load_manifest(banks_dir)
        bank_map = {(b.layer, b.projection): b for b in banks}
        if set(bank_map) != seen or any(tuple(rec["expert_ids"]) != bank_map[(rec["layer"], rec["projection"])].ids for rec in records):
            raise ValueError("Stage-2A bank selection differs from Stage-2B carrier")
        origin_confirmed = True
    return sorted(records, key=lambda rec: (rec["layer"], rec["projection"])), full, origin_confirmed


def create_binding_plan(checkpoint: Checkpoint, *, nonexpert: Any | None = None,
                        experts: Any | None = None, banks_dir: Path | None = None) -> dict:
    checkpoint.validate_inkling()
    roles = classify_all(checkpoint)
    steps = schedule(checkpoint.config)
    available: list[dict] = []
    all_nonexpert = False
    full_experts = False
    origin_confirmed = False
    banks: list[dict] = []
    refs = {}
    if nonexpert is not None:
        available, all_nonexpert = inspect_nonexpert(nonexpert, checkpoint)
        refs["nonexpert"] = {"path": str(nonexpert.path), "artifact_id": nonexpert.artifact_id.hex(),
                             "storage_contract": STORAGE_CONTRACT}
    if experts is not None:
        banks, full_experts, origin_confirmed = inspect_experts(experts, checkpoint, banks_dir=banks_dir)
        refs["experts"] = {"path": str(experts.path), "artifact_id": experts.artifact_id.hex(),
                           "storage_contract": EXPERT_CONTRACT}
    unresolved = [r.source for r in roles if r.category == "unresolved"]
    metadata_complete = all_nonexpert and full_experts and not unresolved and origin_confirmed
    return {
        "contract": CONTRACT, "runtime_loadable": False,
        "checkpoint": {"source": str(checkpoint.root),
                       "model_index_sha256": checkpoint.index_sha256,
                       "config_sha256": sha256((checkpoint.root / "config.json").read_bytes()).hexdigest(),
                       "tensors": len(checkpoint.spans),
                       "expert_source_tensors": len(checkpoint.expert_names),
                       "nonexpert_source_tensors": len(checkpoint.remaining_names)},
        "coverage": {"nonexpert_bound": len(available),
                     "nonexpert_expected": len(checkpoint.remaining_names),
                     "expert_banks_bound": len(banks), "expert_banks_expected": 48,
                     "all_128_experts_in_each_bank": bool(banks) and full_experts,
                     "bank_manifest_matches": origin_confirmed,
                     "unresolved_roles": unresolved,
                     "metadata_complete": metadata_complete,
                     "weight_bytes_verified": False},
        "role_categories": dict(sorted(Counter(r.category for r in roles).items())),
        "roles": [r.json() for r in roles],
        "bound_nonexperts": available,
        "expert_banks": banks,
        "execution": {"logical_steps": 48, "cache_slots": 48, "steps": steps},
        "carriers": refs,
        "limitations": [
            "Binding manifest only; not a combined NInfer model artifact",
            "Missing native Inkling attention/message/loop/MoE runtime integration",
            "MXFP8 is a raw v3 resource; its GPU GEMV is not connected to ninfer-serve",
            "Separate mtp.safetensors is not included",
            "Manifest checks are metadata/IDs only unless --verify-bytes is supplied",
        ],
    }


def write_plan(path: Path, plan: dict) -> Path:
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"binding output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    # O_EXCL ensures we never clobber a previously generated binding plan.
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path

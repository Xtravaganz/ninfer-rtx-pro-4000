"""Stage 4A: losslessly pack Inkling 2 Mini *non-expert* HF weights into NInfer v3.

This module is deliberately a data/metadata carrier, not an executable model.
It uses stdlib for preflight, bounded safetensors reads and comparison; the real
NInfer v3 ArtifactWriter/Artifact are imported only for pack/verify.

The HF index is authoritative: every listed tensor must match a shard header.
All tensors except the 176 packed expert source tensors are preserved, including
router, loop, attention, norms, text embeddings and main-checkpoint vision.
The separate mtp.safetensors file is NOT part of the HF model index.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import struct
from math import prod
from typing import Iterable

# Standard safetensors dtype names and on-disk widths. Unknown types fail shut.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E5M2": 1,
}
DIRECT_FORMATS = {"BF16": "bf16", "F32": "fp32", "I32": "int32"}
EXPERT = re.compile(
    r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
    r"(?:_scale|_amax|_outer_scale_denominator)?$"
)
STORAGE_CONTRACT = "inkling2_hf_nonexpert_v3_v1"
DEFAULT_CHUNK = 4 * 1024 * 1024
MAX_HEADER = 128 * 1024 * 1024


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON member in checkpoint metadata: {key!r}")
        obj[key] = value
    return obj


def _json_read(raw: bytes) -> object:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON value in checkpoint metadata: {value}")
    return json.loads(raw, object_pairs_hook=_json_object, parse_constant=reject)


@dataclass(frozen=True, slots=True)
class TensorSpan:
    name: str
    shard: Path
    offset: int
    byte_count: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def obj_id(self) -> str:
        return "hf/" + self.name

    @property
    def direct_format(self) -> str | None:
        return DIRECT_FORMATS.get(self.dtype)

    def record(self) -> dict:
        return {
            "name": self.name,
            "object": self.obj_id,
            "kind": "tensor" if self.direct_format else "resource",
            "dtype": self.dtype,
            "shape": list(self.shape),
            "bytes": self.byte_count,
        }


def is_expert_source(name: str) -> bool:
    """Select only the quantized expert tensors already carried by Stage 2A/2B."""
    if ".mlp.experts." not in name:
        return False
    match = EXPERT.fullmatch(name)
    if match is None or not 0 <= int(match.group(1)) < 24:
        raise ValueError(f"unknown Inkling expert tensor: {name}")
    layer = int(match.group(1))
    if layer >= 20 and name.endswith(("_amax", "_outer_scale_denominator")):
        raise ValueError(f"MXFP8 expert unexpectedly has NVFP4 auxiliary tensor: {name}")
    return True


def _safe_shard(name: object) -> str:
    if (
        not isinstance(name, str) or not name or name in (".", "..")
        or Path(name).name != name or "/" in name or "\\" in name or "\0" in name
    ):
        raise ValueError(f"unsafe checkpoint shard name: {name!r}")
    return name


def _span_from_header(name: str, record: object, path: Path, payload_start: int,
                      file_size: int) -> TensorSpan:
    if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
        raise ValueError(f"{name}: unsupported safetensors tensor descriptor")
    dtype = record["dtype"]
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"{name}: unsupported safetensors dtype {dtype!r}")
    raw_shape = record["shape"]
    if (not isinstance(raw_shape, list) or len(raw_shape) > 16 or
            any(type(dim) is not int or dim <= 0 for dim in raw_shape)):
        raise ValueError(f"{name}: invalid safetensors shape {raw_shape!r}")
    bounds = record["data_offsets"]
    if (not isinstance(bounds, list) or len(bounds) != 2 or
            any(type(v) is not int for v in bounds)):
        raise ValueError(f"{name}: invalid offsets")
    begin, end = bounds
    expected = prod(raw_shape) * DTYPE_BYTES[dtype]
    if expected <= 0 or not 0 <= begin <= end or end - begin != expected:
        raise ValueError(f"{name}: byte count disagrees with shape/dtype")
    if payload_start + end > file_size:
        raise ValueError(f"{name}: data exceeds shard size (incomplete download?)")
    return TensorSpan(name, path, payload_start + begin, expected, dtype, tuple(raw_shape))


class Checkpoint:
    """Preflight the *entire* indexed main checkpoint, but never load weight tensors."""

    def __init__(self, source: Path):
        self.root = Path(source).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        cfg = self.root / "config.json"
        index_path = self.root / "model.safetensors.index.json"
        self.config = _json_read(cfg.read_bytes())
        self.index_bytes = index_path.read_bytes()
        self.index_sha256 = sha256(self.index_bytes).hexdigest()
        index = _json_read(self.index_bytes)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json lacks a nonempty weight_map")
        if any(not isinstance(k, str) or not k or "\0" in k for k in weight_map):
            raise ValueError("invalid tensor name in checkpoint index")
        paths: dict[str, set[str]] = {}
        for tensor_name, shard_name in weight_map.items():
            shard = _safe_shard(shard_name)
            paths.setdefault(shard, set()).add(tensor_name)
        self.spans: dict[str, TensorSpan] = {}
        self._fds: dict[Path, int] = {}
        for shard_name, expected_names in sorted(paths.items()):
            path = self.root / shard_name
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint shard: {path}")
            file_size = path.stat().st_size
            with path.open("rb") as f:
                length_data = f.read(8)
                if len(length_data) != 8:
                    raise ValueError(f"{path}: truncated safetensors prefix")
                header_length, = struct.unpack("<Q", length_data)
                if not 0 < header_length <= MAX_HEADER or 8 + header_length > file_size:
                    raise ValueError(f"{path}: invalid safetensors header size")
                raw = f.read(header_length)
                if len(raw) != header_length:
                    raise ValueError(f"{path}: truncated safetensors JSON")
            header = _json_read(raw)
            if not isinstance(header, dict):
                raise ValueError(f"{path}: invalid safetensors JSON")
            names = set(header) - {"__metadata__"}
            missing, extra = expected_names - names, names - expected_names
            if missing or extra:
                raise ValueError(
                    f"{path.name}: header/index mismatch: missing={sorted(missing)[:4]}, "
                    f"extra={sorted(extra)[:4]}"
                )
            previous = 0
            shard_spans = []
            for name in sorted(names):
                span = _span_from_header(name, header[name], path,
                                         8 + header_length, file_size)
                shard_spans.append(span)
                self.spans[name] = span
            # Detect overlapping tensor ranges, even for excluded expert tensors.
            for span in sorted(shard_spans, key=lambda x: x.offset):
                begin = span.offset - 8 - header_length
                if begin < previous:
                    raise ValueError(f"{path.name}: overlapping tensor data at {span.name}")
                previous = begin + span.byte_count
        self.expert_names = tuple(sorted(k for k in self.spans if is_expert_source(k)))
        self.remaining_names = tuple(sorted(set(self.spans) - set(self.expert_names)))
        if not self.remaining_names:
            raise ValueError("main checkpoint contains no non-expert tensors")

    def validate_inkling(self) -> None:
        """Strictly check official model structure and expected 176 expert tensors."""
        # Imported by name to work both in the standalone CLI and existing tests.
        from check_checkpoint import validate_config, validate_index, expected_parameters
        validate_config(self.config)
        validate_index({"weight_map": {name: span.shard.name for name, span in self.spans.items()}})
        expected_experts = {name for name in expected_parameters() if ".mlp.experts." in name}
        actual_experts = set(self.expert_names)
        if expected_experts != actual_experts:
            raise ValueError(
                f"expected exactly {len(expected_experts)} expert tensors, "
                f"missing={sorted(expected_experts - actual_experts)[:4]}, "
                f"extra={sorted(actual_experts - expected_experts)[:4]}"
            )
        # The original Stage-1 validator also checks the quantized expert shapes.
        for name, (dtype, shape) in expected_parameters().items():
            span = self.spans[name]
            if span.dtype != dtype or span.shape != tuple(shape):
                raise ValueError(
                    f"{name}: expected {dtype} {tuple(shape)}, "
                    f"got {span.dtype} {span.shape}"
                )

    def select(self, patterns: tuple[str, ...] = ()) -> tuple[TensorSpan, ...]:
        if not patterns:
            return tuple(self.spans[k] for k in self.remaining_names)
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("empty --include selection")
            if not any(fnmatchcase(name, pattern) for name in self.remaining_names):
                raise ValueError(f"--include pattern matches no non-expert tensor: {pattern}")
        return tuple(self.spans[name] for name in self.remaining_names
                     if any(fnmatchcase(name, pat) for pat in patterns))

    def read(self, span: TensorSpan, begin: int, length: int) -> bytes:
        if not 0 <= begin <= begin + length <= span.byte_count:
            raise ValueError(f"{span.name}: out-of-bounds tensor read")
        if span.shard not in self._fds:
            self._fds[span.shard] = os.open(span.shard, os.O_RDONLY)
        result = os.pread(self._fds[span.shard], length, span.offset + begin)
        if len(result) != length:
            raise ValueError(f"{span.name}: truncated tensor data")
        return result

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def payload_estimate(spans: Iterable[TensorSpan]) -> int:
    """Exact planned logical NInfer v3 payload bytes, including 256B padding."""
    cursor = 0
    for span in spans:
        if span.direct_format:
            cursor = (cursor + 255) // 256 * 256
        cursor += span.byte_count
    return cursor


def describe_plan(spans: tuple[TensorSpan, ...]) -> dict:
    counts = {"bf16": 0, "fp32": 0, "int32": 0, "raw": 0}
    for span in spans:
        counts[span.direct_format or "raw"] += 1
    return {
        "tensors": len(spans),
        "total_bytes": sum(s.byte_count for s in spans),
        "payload_bytes": payload_estimate(spans),
        "formats": counts,
    }


def _component_config(cfg: dict) -> dict:
    text = dict(cfg["text_config"])
    text.update(architectures=cfg["architectures"],
                model_type="inkling2_loop_text", eos_token_id=cfg["eos_token_id"])
    return text


def _write_specs(spans: tuple[TensorSpan, ...]):
    from tools.artifact.schema import ResourceSpec, TensorSpec
    specs = []
    bindings = {}
    resources = {}
    for span in spans:
        if span.direct_format:
            specs.append(TensorSpec(span.obj_id, span.shape, span.direct_format,
                                    "contiguous_le_v1"))
            # HF parameter names are retained without implying a runnable NInfer model.
            bindings[span.name] = {"object": span.obj_id}
        else:
            specs.append(ResourceSpec(span.obj_id, span.byte_count, "raw_bytes_v1"))
            resources[span.obj_id] = span.obj_id
    return specs, bindings, resources


def pack_remaining(checkpoint: Checkpoint, destination: Path, *,
                   includes: tuple[str, ...] = (),
                   require_complete: bool = False,
                   chunk_bytes: int = DEFAULT_CHUNK,
                   max_file_bytes: int | None = None) -> dict:
    """Create a NInfer-v3 non-expert carrier. Failure must not publish output."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    spans = checkpoint.select(includes)
    if require_complete and len(spans) != len(checkpoint.remaining_names):
        raise ValueError("--require-complete needs all indexed non-expert tensors")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination == checkpoint.root or checkpoint.root in destination.parents:
        raise ValueError("refusing to write an artifact inside the HF checkpoint")
    from tools.artifact.writer import ArtifactWriter
    specs, bindings, resources = _write_specs(spans)
    components = {"text": {"config": _component_config(checkpoint.config),
                           "resources": resources}}
    if "vision_config" in checkpoint.config:
        components["vision"] = {"config": checkpoint.config["vision_config"]}
    metadata = {
        "name": "Inkling 2 Mini - nonexpert HF weights (not executable)",
        "stage": "4A",
        "storage_contract": STORAGE_CONTRACT,
        "runtime_loadable": False,
        "main_checkpoint_only": True,
        "tensors": [s.record() for s in spans],
    }
    provenance = {
        "model_index_sha256": checkpoint.index_sha256,
        "hf_architectures": checkpoint.config["architectures"],
        "expert_bank_source": "Stage2A/2B separate carrier; not embedded",
        "mtp": "separate mtp.safetensors not included",
        "data": "byte-for-byte copied; no dtype change or transposition",
    }
    kw = {} if max_file_bytes is None else {"max_file_bytes": max_file_bytes}
    with ArtifactWriter(destination, specs, components=components,
                        bindings=bindings, uses=[], metadata=metadata,
                        provenance=provenance, **kw) as writer:
        for span in spans:
            for begin in range(0, span.byte_count, chunk_bytes):
                raw = checkpoint.read(span, begin, min(chunk_bytes, span.byte_count - begin))
                writer.write_region(span.obj_id, begin, raw)
    return {**describe_plan(spans), "output": str(destination)}


def verify_remaining(checkpoint: Checkpoint, destination: Path, *,
                     includes: tuple[str, ...] = (),
                     require_complete: bool = False,
                     chunk_bytes: int = DEFAULT_CHUNK) -> dict:
    """Compare each source tensor's exact bytes to the native NInfer v3 output."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    from tools.artifact.reader import Artifact
    with Artifact(destination) as artifact:
        meta = artifact.directory.metadata
        if (meta.get("storage_contract") != STORAGE_CONTRACT or
                meta.get("runtime_loadable") is not False or
                meta.get("main_checkpoint_only") is not True):
            raise ValueError("not a Stage-4A non-expert NInfer-v3 carrier")
        if artifact.directory.provenance.get("model_index_sha256") != checkpoint.index_sha256:
            raise ValueError("checkpoint index digest differs from packed artifact")
        records = meta.get("tensors")
        if not isinstance(records, list) or not records:
            raise ValueError("artifact has no non-expert tensor manifest")
        names = [r.get("name") for r in records if isinstance(r, dict)]
        if len(names) != len(records) or len(set(names)) != len(names):
            raise ValueError("duplicate/malformed tensor record")
        if any(not isinstance(name, str) or name not in checkpoint.remaining_names for name in names):
            raise ValueError("artifact contains unknown/non-nonexpert HF tensor")
        if includes and set(names) != {s.name for s in checkpoint.select(includes)}:
            raise ValueError("artifact tensor selection does not match --include")
        if require_complete and set(names) != set(checkpoint.remaining_names):
            raise ValueError("incomplete Stage-4A artifact")
        if len(artifact.objects) != len(records) or set(artifact.by_id) != {"hf/" + n for n in names}:
            raise ValueError("artifact object set does not match tensor manifest")
        for record in records:
            name = record["name"]
            span = checkpoint.spans[name]
            if record != span.record():
                raise ValueError(f"{name}: HF dtype, shape or byte count changed")
            obj = artifact.object(span.obj_id)
            if obj.bytes != span.byte_count:
                raise ValueError(f"{name}: artifact bytes mismatch")
            if span.direct_format:
                if (getattr(obj, "kind", None) != "tensor" or
                        obj.format != span.direct_format or
                        obj.layout != "contiguous_le_v1" or
                        obj.shape != span.shape or
                        artifact.directory.bindings.get(name) != {"object": span.obj_id}):
                    raise ValueError(f"{name}: native tensor layout/binding mismatch")
            else:
                if (getattr(obj, "kind", None) != "resource" or
                        obj.encoding != "raw_bytes_v1" or
                        artifact.directory.components["text"].get("resources", {}).get(span.obj_id) != span.obj_id):
                    raise ValueError(f"{name}: resource layout mismatch")
            for begin in range(0, span.byte_count, chunk_bytes):
                amount = min(chunk_bytes, span.byte_count - begin)
                original = checkpoint.read(span, begin, amount)
                actual = artifact.read_range(obj.offset + begin, amount)
                if actual != original:
                    raise ValueError(f"{name}: byte mismatch at source offset {begin}")
    return {**describe_plan(tuple(checkpoint.spans[n] for n in names)),
            "output": str(destination)}

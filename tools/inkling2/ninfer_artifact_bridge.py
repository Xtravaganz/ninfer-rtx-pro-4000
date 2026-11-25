"""Stage 2B: losslessly bridge staged Inkling2 experts into a NInfer v3 carrier.

NVFP4 is stored as *registered* NInfer batched NVFP4 tensors (native scale
swizzle plus one FP32 divisor per expert). The source FP32 amax/denominator
words are also preserved verbatim as resource objects; the derived FP32
weight divisor may differ by final rounding from exact source arithmetic.

MXFP8 G32 UE8M0 is not a registered NInfer tensor type. Store code words and
scale words as separate raw_bytes_v1 resources, with the source expert IDs.
This is a NInfer v3 artifact *container*, not a runtime-loadable model.

No torch or numpy dependency; temporary buffers are bounded by one 128-row
NVFP4 scale tile and the selected I/O chunk size. Imports of the official
NInfer writer/reader happen only when generating/verifying an actual artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO, Iterator

from native_experts import (
    FMT_MXFP8, FMT_NVFP4, HEADER, MiniGeometry, PROJECTIONS,
    _BAD_E4M3_SCALE, _BAD_FP8, _BAD_UE8M0, format_for_layer,
    read_bank_header,
)

LAYOUT = "inkling2_ninfer_v3_expert_carrier_v1"
NATIVE_NVFP4_LAYOUT = "batched_block_scale_k16_m128x4_v1"
RESOURCE_ENCODING = "raw_bytes_v1"
CHUNK = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Bank:
    path: Path
    layer: int
    projection: str
    fmt: int
    ids: tuple[int, ...]
    rows: int
    cols: int
    code_stride: int
    scale_stride: int
    extra: int
    record_stride: int
    header_bytes: int
    digest: str

    @property
    def prefix(self) -> str:
        return f"inkling2/layer{self.layer:02d}/{self.projection}"

    @property
    def ids_object(self) -> str:
        return self.prefix + "/expert_ids"

    @property
    def tensor_object(self) -> str:
        return self.prefix + "/nvfp4"

    @property
    def scalars_object(self) -> str:
        return self.prefix + "/original_amax_denominator_fp32"

    @property
    def code_object(self) -> str:
        return self.prefix + "/mxfp8_codes"

    @property
    def scale_object(self) -> str:
        return self.prefix + "/mxfp8_ue8m0_scales"


def load_manifest(source: Path, *, geometry: MiniGeometry = MiniGeometry()) -> tuple[Bank, ...]:
    source = Path(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or manifest.get("layout") != "inkling2_native_expert_bank_v1":
        raise ValueError("expected Inkling Stage-2A bank manifest v1")
    entries = manifest.get("banks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Stage-2A manifest has no expert banks")
    result: list[Bank] = []
    seen: set[tuple[int, str]] = set()
    for entry in entries:
        name = entry.get("file", "")
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("unsafe Stage-2A bank filename")
        path = source / name
        header = read_bank_header(path)
        layer, projection = header["layer"], header["projection"]
        key = (layer, projection)
        if key in seen:
            raise ValueError(f"duplicate bank for {key}")
        seen.add(key)
        if not 0 <= layer < 24 or header["format_id"] != format_for_layer(layer):
            raise ValueError("bank format differs from Inkling layer precision")
        if header["rows"] != geometry.rows or header["cols"] != geometry.cols(projection):
            raise ValueError("bank geometry differs from Inkling model")
        ids = tuple(header["expert_ids"])
        if any(expert >= geometry.experts for expert in ids):
            raise ValueError("expert ID exceeds model geometry")
        if (entry.get("layer"), entry.get("projection"), entry.get("expert_ids")) != (
            layer, projection, list(ids)
        ):
            raise ValueError("manifest bank identity differs from binary header")
        if entry.get("bytes") != path.stat().st_size:
            raise ValueError("manifest bank byte size differs from file")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid manifest SHA256")
        result.append(Bank(path, layer, projection, header["format_id"], ids,
                           header["rows"], header["cols"], header["code_stride"],
                           header["scale_stride"], header["extra"],
                           header["record_stride"], header["header_bytes"], digest))
    return tuple(sorted(result, key=lambda b: (b.layer, PROJECTIONS[b.projection])))


def validate_selection(banks: tuple[Bank, ...], *, full_required: bool = False) -> None:
    if not banks:
        raise ValueError("no banks selected")
    if full_required:
        keys = {(b.layer, b.projection) for b in banks}
        if len(keys) != 24 * 2 or any(len(b.ids) != 128 or b.ids != tuple(range(128)) for b in banks):
            raise ValueError("full expert carrier requires both projections and all 128 experts in 24 layers")


def _check_words(data: bytes, table: bytes, label: str) -> None:
    if b"\x01" in data.translate(table):
        raise ValueError(f"{label}: invalid quantized word")


def swizzle_nvfp4_128_rows(scales: bytes, cols: int) -> bytes:
    """NInfer v3 K16 M128x4 swizzle, exactly matching codec's reshape/permute.

    Input [128,K/16] row-major. Output order is [K/64,32,4,4] bytes.
    """
    if cols <= 0 or cols % 64 or len(scales) != 128 * cols // 16:
        raise ValueError("NVFP4 swizzle requires one complete 128-row tile and K divisible by 64")
    stride = cols // 16
    source = memoryview(scales)
    out = bytearray(len(scales))
    cursor = 0
    for k_tile in range(cols // 64):
        for r32 in range(32):
            for group in range(4):
                begin = (group * 32 + r32) * stride + k_tile * 4
                out[cursor:cursor + 4] = source[begin:begin + 4]
                cursor += 4
    return bytes(out)


def unswizzle_nvfp4_128_rows(scales: bytes, cols: int) -> bytes:
    if cols <= 0 or cols % 64 or len(scales) != 128 * cols // 16:
        raise ValueError("NVFP4 unswizzle requires a complete 128-row tile")
    stride = cols // 16
    # Stage 6D: exact array transpose replacing ~4,600 Python slice copies
    # per 128-row tile at K=2304. NumPy is optional: the original
    # dependency-free conversion remains valid on a bare host.
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        return (np.frombuffer(scales, dtype=np.uint8)
                .reshape(cols // 64, 32, 4, 4)
                .transpose(2, 1, 0, 3)
                .tobytes())
    out = bytearray(len(scales))
    cursor = 0
    for k_tile in range(cols // 64):
        for r32 in range(32):
            for group in range(4):
                begin = (group * 32 + r32) * stride + k_tile * 4
                out[begin:begin + 4] = scales[cursor:cursor + 4]
                cursor += 4
    return bytes(out)


def _chunks(f: BinaryIO, size: int, chunk_bytes: int) -> Iterator[bytes]:
    left = size
    while left:
        data = f.read(min(chunk_bytes, left))
        if not data:
            raise ValueError("truncated Stage-2A bank")
        yield data
        left -= len(data)


def _copy(f: BinaryIO, total: int, writer, object_id: str, offset: int,
          digest, chunk_bytes: int, *, validation: bytes | None = None, label: str = "") -> None:
    for part in _chunks(f, total, chunk_bytes):
        if validation is not None:
            _check_words(part, validation, label)
        digest.update(part)
        writer.write_region(object_id, offset, part)
        offset += len(part)


def _weight_divisor(raw_scalars: bytes) -> bytes:
    """NInfer expects decoded = E2M1 * E4M3 / divisor; source uses amax / den.

    Original amax and denominator are preserved in an auxiliary resource.
    We only round the derived *divisor* once to FP32; no requantization occurs.
    """
    if len(raw_scalars) != 8:
        raise ValueError("NVFP4 missing original scalar words")
    amax, den = struct.unpack("<ff", raw_scalars)
    if not (math.isfinite(amax) and math.isfinite(den) and amax > 0 and den > 0):
        raise ValueError("NVFP4 invalid original scalar words")
    divisor = den / amax
    if not math.isfinite(divisor) or not 0 < divisor <= 3.4028234663852886e38:
        raise ValueError("NVFP4 derived divisor cannot be represented in FP32")
    packed = struct.pack("<f", divisor)
    if struct.unpack("<f", packed)[0] == 0:
        raise ValueError("NVFP4 derived divisor underflows FP32")
    return packed


def _read_leading(f: BinaryIO, bank: Bank, digest) -> bytes:
    leading = f.read(bank.header_bytes)
    if len(leading) != bank.header_bytes:
        raise ValueError("truncated bank header")
    digest.update(leading)
    return leading[HEADER.size:]


def _write_bank(writer, bank: Bank, *, chunk_bytes: int) -> None:
    count = len(bank.ids)
    digest = sha256()
    with bank.path.open("rb") as f:
        ids_bytes = _read_leading(f, bank, digest)
        if len(ids_bytes) != 4 * count:
            raise ValueError("invalid expert index table")
        writer.write_region(bank.ids_object, 0, ids_bytes)
        if bank.fmt == FMT_NVFP4:
            # Native plane offsets: row-major codes, native-swizzled K16 scales,
            # and exactly one FP32 divisor per expert in bank index order.
            code_plane = count * bank.code_stride
            scale_plane = count * bank.scale_stride
            scale_base = (code_plane + 255) // 256 * 256
            if scale_base != code_plane:
                writer.write_zeros(bank.tensor_object, code_plane, scale_base - code_plane)
            for index in range(count):
                _copy(f, bank.code_stride, writer, bank.tensor_object,
                      index * bank.code_stride, digest, chunk_bytes)
                if bank.rows % 128 or bank.cols % 64:
                    raise ValueError("NInfer NVFP4 requires 128-row / 64-column alignment")
                tile = 128 * bank.cols // 16
                for tile_index in range(bank.rows // 128):
                    raw = f.read(tile)
                    if len(raw) != tile:
                        raise ValueError("truncated NVFP4 scales")
                    digest.update(raw)
                    _check_words(raw, _BAD_E4M3_SCALE, "NVFP4 scales")
                    native_scales = swizzle_nvfp4_128_rows(raw, bank.cols)
                    writer.write_region(bank.tensor_object,
                                        scale_base + index * bank.scale_stride + tile_index * tile,
                                        native_scales)
                scalars = f.read(8)
                if len(scalars) != 8:
                    raise ValueError("truncated NVFP4 scalars")
                digest.update(scalars)
                writer.write_region(bank.scalars_object, index * 8, scalars)
                writer.write_region(bank.tensor_object,
                                    scale_base + scale_plane + index * 4,
                                    _weight_divisor(scalars))
        else:
            for index in range(count):
                _copy(f, bank.code_stride, writer, bank.code_object,
                      index * bank.code_stride, digest, chunk_bytes,
                      validation=_BAD_FP8, label="MXFP8 FP8 codes")
                _copy(f, bank.scale_stride, writer, bank.scale_object,
                      index * bank.scale_stride, digest, chunk_bytes,
                      validation=_BAD_UE8M0, label="MXFP8 UE8M0 scales")
        if f.read(1):
            raise ValueError("bank contains trailing data")
    if digest.hexdigest() != bank.digest:
        raise ValueError(f"{bank.path.name}: SHA256 differs from Stage-2A manifest")


def _bank_specs(banks: tuple[Bank, ...]):
    from tools.artifact.schema import ResourceSpec, TensorSpec
    specs = []
    resources = {}
    bindings = {}
    records = []
    for bank in banks:
        ids_id = bank.ids_object
        specs.append(ResourceSpec(ids_id, len(bank.ids) * 4, RESOURCE_ENCODING))
        resources[bank.prefix + "/expert_ids"] = ids_id
        record = {
            "layer": bank.layer, "projection": bank.projection, "expert_ids": list(bank.ids),
            "shape": [len(bank.ids), bank.rows, bank.cols],
            "quantization": "nvfp4_e2m1_g16_e4m3" if bank.fmt == FMT_NVFP4 else "mxfp8_e4m3_g32_ue8m0",
        }
        if bank.fmt == FMT_NVFP4:
            tensor_id = bank.tensor_object
            specs.append(TensorSpec(tensor_id, (len(bank.ids), bank.rows, bank.cols),
                                    "nvfp4", NATIVE_NVFP4_LAYOUT))
            bindings[tensor_id] = {"object": tensor_id}
            specs.append(ResourceSpec(bank.scalars_object, len(bank.ids) * 8, RESOURCE_ENCODING))
            resources[bank.scalars_object] = bank.scalars_object
            record.update(native_tensor=tensor_id, original_scalars=bank.scalars_object,
                          divisor_policy="fp32(outer_denominator / amax)")
        else:
            specs.append(ResourceSpec(bank.code_object, len(bank.ids) * bank.code_stride, RESOURCE_ENCODING))
            specs.append(ResourceSpec(bank.scale_object, len(bank.ids) * bank.scale_stride, RESOURCE_ENCODING))
            resources[bank.code_object] = bank.code_object
            resources[bank.scale_object] = bank.scale_object
            record.update(codes=bank.code_object, scales=bank.scale_object,
                          scale_group=32, scale_dtype="UE8M0_U8")
        records.append(record)
    return specs, resources, bindings, records


def _component_config(config: dict) -> dict:
    from check_checkpoint import validate_config
    validate_config(config)
    raw = config["text_config"]
    # Contract-only artifact; architecture is *not* registered for execution.
    return {**raw, "architectures": config["architectures"],
            "model_type": "inkling2_loop_text", "eos_token_id": config["eos_token_id"]}


def pack_banks(source: Path, destination: Path, config: dict, *,
               chunk_bytes: int = CHUNK, geometry: MiniGeometry = MiniGeometry(),
               full_required: bool = False, banks: tuple[Bank, ...] | None = None,
               max_file_bytes: int | None = None) -> dict:
    """Create a NInfer-v3-readable artifact; not a runnable Inkling2 model."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("output artifact must not be inside bank source")
    if destination.exists():
        raise FileExistsError(destination)
    banks = load_manifest(source, geometry=geometry) if banks is None else banks
    validate_selection(banks, full_required=full_required)
    from tools.artifact.writer import ArtifactWriter
    specs, resources, bindings, records = _bank_specs(banks)
    kw = {} if max_file_bytes is None else {"max_file_bytes": max_file_bytes}
    with ArtifactWriter(destination, specs,
                        components={"text": {"config": _component_config(config),
                                             "resources": resources}},
                        bindings=bindings,
                        uses=[],
                        metadata={"name": "Inkling 2 Mini - expert weight carrier (not executable)",
                                  "stage": "2B", "inkling2_banks": records,
                                  "storage_contract": LAYOUT,
                                  "runtime_loadable": False},
                        provenance={"stage2a_manifest_sha256": sha256(
                            (source / "manifest.json").read_bytes()).hexdigest(),
                                    "input_stage": "Inkling 2 Mini Stage 2A",
                                    "nvfp4_derived_divisor_warning":
                                      "source FP32 amax and denominator are preserved; reciprocal rounds to FP32",
                                    "mxfp8": "stored as raw_bytes_v1 resources until registered runtime format"},
                        **kw) as writer:
        for bank in banks:
            _write_bank(writer, bank, chunk_bytes=chunk_bytes)
    return {"artifact": str(destination), "banks": len(banks),
            "native_nvfp4_banks": sum(b.fmt == FMT_NVFP4 for b in banks),
            "opaque_mxfp8_banks": sum(b.fmt == FMT_MXFP8 for b in banks)}


def verify_artifact(source: Path, artifact_path: Path, *,
                    geometry: MiniGeometry = MiniGeometry(),
                    chunk_bytes: int = CHUNK) -> dict:
    """Bit-compare the carrier's code/scales/resources against Stage-2A banks."""
    from tools.artifact.reader import Artifact
    banks = load_manifest(source, geometry=geometry)
    with Artifact(artifact_path) as artifact:
        if artifact.directory.provenance.get("stage2a_manifest_sha256") != sha256(
            (Path(source) / "manifest.json").read_bytes()
        ).hexdigest():
            raise ValueError("artifact provenance differs from Stage-2A manifest")
        meta = artifact.directory.metadata
        if meta.get("storage_contract") != LAYOUT or meta.get("runtime_loadable") is not False:
            raise ValueError("artifact is not an Inkling2 Stage-2B carrier")
        records = meta.get("inkling2_banks", [])
        wanted = {(b.layer, b.projection): b for b in banks}
        if len(records) != len(wanted):
            raise ValueError("artifact and bank manifest contain different bank counts")
        for record in records:
            key = (record.get("layer"), record.get("projection"))
            if key not in wanted:
                raise ValueError("unexpected artifact bank")
            bank = wanted.pop(key)
            # Source authenticity is independent of comparing two mutable copies.
            source_digest = sha256()
            with bank.path.open("rb") as original:
                for part in _chunks(original, bank.path.stat().st_size, chunk_bytes):
                    source_digest.update(part)
            if source_digest.hexdigest() != bank.digest:
                raise ValueError("Stage-2A bank SHA256 differs from manifest")
            if record.get("expert_ids") != list(bank.ids):
                raise ValueError("artifact expert-ID metadata mismatch")
            with bank.path.open("rb") as f:
                f.seek(bank.header_bytes)
                if artifact.read_object(bank.ids_object) != struct.pack(f"<{len(bank.ids)}I", *bank.ids):
                    raise ValueError("artifact expert IDs differ")
                if bank.fmt == FMT_NVFP4:
                    tensor = artifact.object(bank.tensor_object)
                    count = len(bank.ids)
                    code_bytes = count * bank.code_stride
                    scale_base = (code_bytes + 255) // 256 * 256
                    for index in range(count):
                        cursor = 0
                        for part in _chunks(f, bank.code_stride, chunk_bytes):
                            actual = artifact.read_range(tensor.offset + index * bank.code_stride + cursor, len(part))
                            if part != actual:
                                raise ValueError("NVFP4 weight codes changed")
                            cursor += len(part)
                        tile = 128 * bank.cols // 16
                        for tile_index in range(bank.rows // 128):
                            scales = f.read(tile)
                            output = artifact.read_range(tensor.offset + scale_base +
                                                         index * bank.scale_stride + tile_index * tile, tile)
                            if scales != unswizzle_nvfp4_128_rows(output, bank.cols):
                                raise ValueError("NVFP4 weight scales changed")
                        original_scalars = f.read(8)
                        aux = artifact.read_range(artifact.object(bank.scalars_object).offset + index * 8, 8)
                        if original_scalars != aux:
                            raise ValueError("NVFP4 original scalar words changed")
                        divisor = artifact.read_range(tensor.offset + scale_base + count*bank.scale_stride + index*4, 4)
                        if divisor != _weight_divisor(original_scalars):
                            raise ValueError("NVFP4 derived divisor mismatch")
                else:
                    codes = artifact.object(bank.code_object)
                    scales = artifact.object(bank.scale_object)
                    for index in range(len(bank.ids)):
                        cursor = 0
                        for part in _chunks(f, bank.code_stride, chunk_bytes):
                            actual = artifact.read_range(codes.offset + index * bank.code_stride + cursor, len(part))
                            if part != actual:
                                raise ValueError("MXFP8 weight codes changed")
                            cursor += len(part)
                        cursor = 0
                        for part in _chunks(f, bank.scale_stride, chunk_bytes):
                            actual = artifact.read_range(scales.offset + index * bank.scale_stride + cursor, len(part))
                            if part != actual:
                                raise ValueError("MXFP8 UE8M0 scales changed")
                            cursor += len(part)
                if f.read(1):
                    raise ValueError("trailing data in source bank")
        if wanted:
            raise ValueError("missing artifact bank")
        return {"banks": len(banks), "native_nvfp4": sum(b.fmt == FMT_NVFP4 for b in banks),
                "mxfp8_resources": sum(b.fmt == FMT_MXFP8 for b in banks),
                "artifact_id": artifact.artifact_id.hex()}

"""Inkling 2 Mini native expert weights: read-only safetensors -> lossless staged bank.

Stdlib only. The bank format is a transport/staging contract; NInfer cannot load
these files as its registered artifact format without a future runtime/codec PR.

Layout v1:
    fixed 64-byte header, little-endian: <8s9I2QI
    expert_indices: count * uint32 (sorted, exact original IDs)
    each expert in index order:
        codes:  rows * (cols//2) bytes NVFP4, or rows*cols bytes MXFP8
        scales: rows*(cols//16) bytes E4M3 NVFP4, or rows*(cols//32) bytes UE8M0
        NVFP4 only: amax FP32, outer_scale_denominator FP32 (raw words)

Values are copied bit-for-bit from original checkpoint. On-disk source axes
[expert, output_row, input_feature] are preserved. NVFP4 scale bytes are
natural row-major (NOT the swizzled NInfer weight layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import BinaryIO, Callable, Iterator

MAGIC = b"I2EXPERT"
VERSION = 1
FMT_NVFP4, FMT_MXFP8 = 1, 2
PROJECTIONS = {"w13": 1, "w2": 2}
HEADER = struct.Struct("<8s9I2QI")  # exactly 64 bytes
assert HEADER.size == 64
FP32 = struct.Struct("<f")

# Precomputed C-level translation for fast byte-word validation.
_BAD_E4M3_SCALE = bytes(int((i & 128) != 0 or i == 127) for i in range(256))
_BAD_FP8 = bytes(int((i & 127) == 127) for i in range(256))
_BAD_UE8M0 = bytes(int(i == 255) for i in range(256))


@dataclass(frozen=True)
class MiniGeometry:
    experts: int = 128
    rows: int = 2304
    gate_up_cols: int = 2304
    down_cols: int = 1152

    def cols(self, projection: str) -> int:
        if projection not in PROJECTIONS:
            raise ValueError(f"invalid expert projection {projection}")
        return self.gate_up_cols if projection == "w13" else self.down_cols


@dataclass(frozen=True)
class TensorSpan:
    file: Path
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]


class IndexedSafetensors:
    """Lazy bounded pread of named tensors; never allocates full tensors."""

    def __init__(self, source: Path):
        self.root = Path(source)
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"expected {index_path}")
        index = json.loads(index_path.read_text())
        if not isinstance(index.get("weight_map"), dict):
            raise ValueError("invalid checkpoint index")
        self.weight_map = index["weight_map"]
        self._headers: dict[Path, dict] = {}
        self._header_sizes: dict[Path, int] = {}
        self._open: dict[Path, int] = {}

    def _header(self, path: Path) -> dict:
        if path in self._headers:
            return self._headers[path]
        with path.open("rb") as f:
            header_size_raw = f.read(8)
            if len(header_size_raw) != 8:
                raise ValueError(f"{path}: missing header size")
            header_size, = struct.unpack("<Q", header_size_raw)
            if not 0 < header_size <= 128 * 1024 * 1024:
                raise ValueError(f"{path}: unsafe header size")
            if header_size + 8 > os.fstat(f.fileno()).st_size:
                raise ValueError(f"{path}: header exceeds file size")
            header = json.loads(f.read(header_size))
            if not isinstance(header, dict):
                raise ValueError(f"{path}: invalid JSON header")
        self._headers[path] = header
        self._header_sizes[path] = header_size
        return header

    def describe(self, name: str) -> TensorSpan:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise ValueError(f"missing checkpoint tensor {name}") from exc
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"unsafe shard name for {name}")
        path = self.root / shard
        header = self._header(path)
        try:
            value = header[name]
            shape = tuple(value["shape"])
            dtype = value["dtype"]
            begin, end = value["data_offsets"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid tensor header for {name}") from exc
        if dtype not in {"U8", "F8_E4M3", "F32"}:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        if not shape or any(type(i) is not int or i < 0 for i in shape):
            raise ValueError(f"invalid shape for {name}")
        if not isinstance(begin, int) or not isinstance(end, int):
            raise ValueError(f"invalid offsets for {name}")
        byte_size = 4 if dtype == "F32" else 1
        expected = math.prod(shape) * byte_size
        if not 0 <= begin <= end or end - begin != expected:
            raise ValueError(f"{name}: tensor offsets disagree with shape/dtype")
        file_size = path.stat().st_size
        absolute = 8 + self._header_sizes[path] + begin
        if absolute + expected > file_size:
            raise ValueError(f"{name}: tensor extends past shard end")
        return TensorSpan(path, absolute, expected, dtype, shape)

    def read_bytes(self, span: TensorSpan, offset: int, size: int) -> bytes:
        if not 0 <= offset <= offset + size <= span.size:
            raise ValueError("tensor read exceeds declared span")
        if span.file not in self._open:
            self._open[span.file] = os.open(span.file, os.O_RDONLY)
        result = os.pread(self._open[span.file], size, span.offset + offset)
        if len(result) != size:
            raise ValueError(f"short safetensors read from {span.file}")
        return result

    def close(self) -> None:
        for fd in self._open.values():
            os.close(fd)
        self._open.clear()

    def __enter__(self) -> IndexedSafetensors:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def parse_selection(spec: str, limit: int) -> tuple[int, ...]:
    """All, individual numbers, or inclusive ranges like 0,4,20-23."""
    if spec.strip().lower() == "all":
        return tuple(range(limit))
    found: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty selection")
        ends = part.split("-")
        if len(ends) == 1:
            a = b = int(ends[0])
        elif len(ends) == 2:
            a, b = (int(n) for n in ends)
        else:
            raise ValueError(f"invalid range {part}")
        if not 0 <= a <= b < limit:
            raise ValueError(f"selection {part} outside [0,{limit})")
        found.update(range(a, b + 1))
    if not found:
        raise ValueError("empty selection")
    return tuple(sorted(found))


def format_for_layer(layer: int) -> int:
    if not 0 <= layer < 24:
        raise ValueError("Inkling Mini has physical layers 0..23")
    return FMT_NVFP4 if layer < 20 else FMT_MXFP8


@dataclass(frozen=True)
class ExpertTensors:
    layer: int
    projection: str
    format_id: int
    rows: int
    cols: int
    weight: TensorSpan
    scales: TensorSpan
    amax: TensorSpan | None
    denominator: TensorSpan | None

    @property
    def code_stride(self) -> int:
        return self.rows * (self.cols // 2 if self.format_id == FMT_NVFP4 else self.cols)

    @property
    def scale_stride(self) -> int:
        return self.rows * (self.cols // (16 if self.format_id == FMT_NVFP4 else 32))

    @property
    def record_stride(self) -> int:
        return self.code_stride + self.scale_stride + (8 if self.format_id == FMT_NVFP4 else 0)

    def name(self) -> str:
        return f"model.llm.layers.{self.layer}.mlp.experts.{self.projection}_weight"


def describe_expert_bank(store: IndexedSafetensors, layer: int, projection: str,
                         geometry: MiniGeometry = MiniGeometry()) -> ExpertTensors:
    fmt = format_for_layer(layer)
    n, k, e = geometry.rows, geometry.cols(projection), geometry.experts
    if n <= 0 or e <= 0 or k <= 0 or (k % 32) != 0:
        raise ValueError("invalid expert-bank dimensions")
    prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
    weight = store.describe(prefix)
    scale = store.describe(prefix + "_scale")
    expected_weight = (e, n, k // 2 if fmt == FMT_NVFP4 else k)
    expected_scales = (e, n, k // (16 if fmt == FMT_NVFP4 else 32))
    expected_type = "U8" if fmt == FMT_NVFP4 else "F8_E4M3"
    scale_type = "F8_E4M3" if fmt == FMT_NVFP4 else "U8"
    if weight.shape != expected_weight or weight.dtype != expected_type:
        raise ValueError(f"{prefix}: expected {expected_type} {expected_weight}, got {weight.dtype} {weight.shape}")
    if scale.shape != expected_scales or scale.dtype != scale_type:
        raise ValueError(f"{prefix}_scale: expected {scale_type} {expected_scales}, got {scale.dtype} {scale.shape}")
    amax = denominator = None
    if fmt == FMT_NVFP4:
        amax = store.describe(prefix + "_amax")
        denominator = store.describe(prefix + "_outer_scale_denominator")
        if amax.dtype != "F32" or amax.shape != (e,):
            raise ValueError(f"{prefix}_amax: expected FP32 [{e}]")
        if denominator.dtype != "F32" or denominator.shape != (1,):
            raise ValueError(f"{prefix}_outer_scale_denominator: expected FP32 [1]")
    return ExpertTensors(layer, projection, fmt, n, k, weight, scale, amax, denominator)


def float32_word(raw: bytes, name: str) -> float:
    if len(raw) != 4:
        raise ValueError(f"{name}: missing scalar")
    value, = FP32.unpack(raw)
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name}: expected positive finite F32")
    return value


def _check_words(words: bytes, table: bytes, message: str) -> None:
    if b"\x01" in words.translate(table):
        raise ValueError(message)


def _copy_region(store: IndexedSafetensors, src: TensorSpan, src_offset: int,
                 size: int, dst: BinaryIO, digest, chunk_bytes: int,
                 table: bytes | None = None, label: str = "") -> None:
    for offset in range(0, size, chunk_bytes):
        data = store.read_bytes(src, src_offset + offset, min(chunk_bytes, size - offset))
        if table is not None:
            _check_words(data, table, f"{label}: invalid quantization word")
        dst.write(data)
        digest.update(data)


def export_bank(store: IndexedSafetensors, bank: ExpertTensors, experts: tuple[int, ...],
                output: Path, chunk_bytes: int = 4 * 1024 * 1024,
                force: bool = False) -> dict:
    """Atomic lossless export of one bank. Return manifest entry after success."""
    if not experts or tuple(sorted(set(experts))) != experts or any(
        e < 0 or e >= bank.weight.shape[0] for e in experts
    ):
        raise ValueError("selected experts must be unique, sorted and in range")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if output.exists() and not force:
        raise FileExistsError(f"refusing to replace {output}; pass --force explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    code_table = _BAD_FP8 if bank.format_id == FMT_MXFP8 else None
    scale_table = _BAD_E4M3_SCALE if bank.format_id == FMT_NVFP4 else _BAD_UE8M0
    digest = sha256()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=output.name + ".", suffix=".part",
                                         dir=output.parent, delete=False) as f:
            temp_path = Path(f.name)
            hdr = HEADER.pack(MAGIC, VERSION, bank.format_id, bank.layer,
                              PROJECTIONS[bank.projection], len(experts), bank.rows,
                              bank.cols, 16 if bank.format_id == FMT_NVFP4 else 32,
                              0, bank.code_stride, bank.scale_stride,
                              8 if bank.format_id == FMT_NVFP4 else 0)
            f.write(hdr)
            digest.update(hdr)
            ids = struct.pack(f"<{len(experts)}I", *experts)
            f.write(ids)
            digest.update(ids)
            for expert in experts:
                _copy_region(store, bank.weight, expert * bank.code_stride, bank.code_stride,
                             f, digest, chunk_bytes, code_table, "FP8 codes")
                _copy_region(store, bank.scales, expert * bank.scale_stride, bank.scale_stride,
                             f, digest, chunk_bytes, scale_table, "expert scales")
                if bank.format_id == FMT_NVFP4:
                    assert bank.amax is not None and bank.denominator is not None
                    amax = store.read_bytes(bank.amax, expert * 4, 4)
                    den = store.read_bytes(bank.denominator, 0, 4)
                    float32_word(amax, "amax")
                    float32_word(den, "outer_scale_denominator")
                    f.write(amax + den)
                    digest.update(amax + den)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return {
        "file": output.name,
        "sha256": digest.hexdigest(),
        "bytes": output.stat().st_size,
        "layer": bank.layer,
        "projection": bank.projection,
        "format": "nvfp4_e2m1_g16_e4m3" if bank.format_id == FMT_NVFP4 else "mxfp8_e4m3_g32_ue8m0",
        "expert_ids": list(experts),
        "rows": bank.rows,
        "cols": bank.cols,
        "code_bytes_per_expert": bank.code_stride,
        "scale_bytes_per_expert": bank.scale_stride,
        "scalar_bytes_per_expert": 8 if bank.format_id == FMT_NVFP4 else 0,
    }


def read_bank_header(path: Path) -> dict:
    with Path(path).open("rb") as f:
        raw = f.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("truncated expert bank header")
        magic, version, fmt, layer, proj, count, rows, cols, group, flags, code, scales, extra = HEADER.unpack(raw)
        if magic != MAGIC or version != VERSION or fmt not in (1, 2) or proj not in (1, 2):
            raise ValueError("unsupported expert bank header")
        if not 0 < count <= 128 or rows <= 0 or cols <= 0 or flags != 0:
            raise ValueError("invalid expert bank geometry")
        if group != (16 if fmt == 1 else 32) or code != rows * (cols // (2 if fmt == 1 else 1)):
            raise ValueError("invalid expert bank code stride")
        if cols % group or scales != rows * (cols // group) or extra != (8 if fmt == 1 else 0):
            raise ValueError("invalid expert bank scale stride")
        id_bytes = f.read(count * 4)
        if len(id_bytes) != count * 4:
            raise ValueError("truncated bank expert indices")
        ids = struct.unpack(f"<{count}I", id_bytes)
        if tuple(sorted(set(ids))) != ids or ids[-1] >= 128:
            raise ValueError("invalid expert indices")
    expected = HEADER.size + 4 * count + count * (code + scales + extra)
    if path.stat().st_size != expected:
        raise ValueError("expert bank size mismatch")
    return dict(layer=layer, projection=next(k for k,v in PROJECTIONS.items() if v == proj),
                format_id=fmt, expert_ids=ids, rows=rows, cols=cols,
                code_stride=code, scale_stride=scales, extra=extra,
                record_stride=code+scales+extra, header_bytes=HEADER.size + 4 * count)


def verify_bank(store: IndexedSafetensors, bank: ExpertTensors,
                path: Path, expected_sha256: str | None = None,
                chunk_bytes: int = 4 * 1024 * 1024) -> dict:
    parsed = read_bank_header(path)
    if (parsed["layer"], parsed["projection"], parsed["format_id"],
        parsed["rows"], parsed["cols"]) != (
        bank.layer, bank.projection, bank.format_id, bank.rows, bank.cols
    ):
        raise ValueError("bank header differs from source tensors")
    check_table = _BAD_FP8 if bank.format_id == FMT_MXFP8 else None
    scale_table = _BAD_E4M3_SCALE if bank.format_id == FMT_NVFP4 else _BAD_UE8M0
    digest = sha256()
    with path.open("rb") as f:
        leading = f.read(parsed["header_bytes"])
        digest.update(leading)
        for expert in parsed["expert_ids"]:
            for src, stride, table in (
                (bank.weight, bank.code_stride, check_table),
                (bank.scales, bank.scale_stride, scale_table),
            ):
                for off in range(0, stride, chunk_bytes):
                    size = min(chunk_bytes, stride - off)
                    stored = f.read(size)
                    original = store.read_bytes(src, expert * stride + off, size)
                    if stored != original:
                        raise ValueError(f"layer {bank.layer} {bank.projection} expert {expert}: changed quant words")
                    if table is not None:
                        _check_words(stored, table, "corrupt quantization words")
                    digest.update(stored)
            if bank.format_id == FMT_NVFP4:
                assert bank.amax is not None and bank.denominator is not None
                aux = f.read(8)
                expected = store.read_bytes(bank.amax, expert * 4, 4) + store.read_bytes(bank.denominator, 0, 4)
                if aux != expected:
                    raise ValueError("changed NVFP4 amax/denominator")
                digest.update(aux)
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError("bank sha256 differs from manifest")
    return {"sha256": digest.hexdigest(), "expert_count": len(parsed["expert_ids"])}


def decode_fp8(word: int) -> float:
    """E4M3FN as defined by the checkpoint (0x7f/0xff are NaN)."""
    if not 0 <= word <= 255 or (word & 127) == 127:
        raise ValueError("invalid E4M3FN word")
    sign = -1.0 if word & 128 else 1.0
    exponent, fraction = (word >> 3) & 15, word & 7
    return sign * ((fraction / 8.0) * 2.0**(-6) if exponent == 0
                   else (1.0 + fraction / 8.0) * 2.0**(exponent - 7))


def decode_nvfp4_row(codes: bytes, scales: bytes, amax: float, den: float) -> list[float]:
    if len(codes) * 2 != len(scales) * 16:
        raise ValueError("NVFP4 codes/scales size mismatch")
    if not math.isfinite(amax) or not math.isfinite(den) or amax <= 0 or den <= 0:
        raise ValueError("invalid NVFP4 scalar multipliers")
    base = (0, .5, 1, 1.5, 2, 3, 4, 6)
    values = []
    ratio = amax / den
    for i, byte in enumerate(codes):
        scale_word = scales[i // 8]
        if scale_word & 128 or scale_word == 127:
            raise ValueError("invalid NVFP4 scale")
        scale = decode_fp8(scale_word) * ratio
        for nibble in (byte & 15, byte >> 4):
            values.append((-1 if nibble & 8 else 1) * base[nibble & 7] * scale)
    return values


def decode_mxfp8_row(codes: bytes, scales: bytes) -> list[float]:
    if len(codes) != len(scales) * 32:
        raise ValueError("MXFP8 codes/scales size mismatch")
    if 255 in scales:
        raise ValueError("MXFP8 scale 255 is reserved")
    return [decode_fp8(word) * (2.0**(scales[i // 32] - 127)) for i, word in enumerate(codes)]

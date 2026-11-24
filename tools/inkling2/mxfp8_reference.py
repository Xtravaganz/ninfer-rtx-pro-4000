"""Inkling 2 Mini Stage 3A: independent MXFP8 E4M3FN/G32/UE8M0 oracle.

No quantization, no production GPU kernel, no modification of NInfer v3 registry.
The checkpoint's last-four-layer expert scales are natural row-major [E,N,K/32],
not a cuBLAS/CUTLASS hardware swizzle. Stage-2B preserves those bytes as
raw_bytes_v1 resources. 0x00 UE8M0 is 2**-127; 0xff is NaN, never zero.

The Python CPU oracle is independent from native_experts.decode_mxfp8_row,
while the optional PyTorch implementation performs decoding on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
from typing import Sequence

_BAD_E4M3 = bytes(int((word & 0x7f) == 0x7f) for word in range(256))
_BAD_UE8M0 = bytes(int(word == 0xff) for word in range(256))


def decode_e4m3fn(word: int) -> float:
    """Decode an OCP E4M3FN byte (including negative zero), reject NaN."""
    if type(word) is not int or not 0 <= word < 256 or (word & 0x7f) == 0x7f:
        raise ValueError('invalid E4M3FN weight code')
    e, mantissa = (word >> 3) & 15, word & 7
    magnitude = math.ldexp(float(mantissa), -9) if e == 0 else math.ldexp(float(8 + mantissa), e - 10)
    return math.copysign(magnitude, -1.0 if word & 0x80 else 1.0)


def decode_ue8m0(word: int) -> float:
    """OCP E8M0: all 0..254 are finite, 0 means 2^-127, 255 is NaN."""
    if type(word) is not int or not 0 <= word < 255:
        raise ValueError('invalid UE8M0 block scale (0xff is NaN)')
    return math.ldexp(1.0, word - 127)


def _valid_bytes(data: bytes, table: bytes, what: str) -> None:
    if b'\x01' in data.translate(table):
        raise ValueError(f'{what}: nonfinite MXFP8 word')


@dataclass(frozen=True, slots=True)
class Matrix:
    """A selected expert's contiguous [rows,K] window and its [rows,K/32] scales."""

    rows: int
    cols: int
    codes: bytes
    scales: bytes

    def __post_init__(self):
        if type(self.rows) is not int or type(self.cols) is not int or self.rows < 1 or self.cols < 32 or self.cols % 32:
            raise ValueError('MXFP8 requires positive row count and K divisible by 32')
        if not isinstance(self.codes, bytes) or not isinstance(self.scales, bytes):
            raise TypeError('MXFP8 input words must be immutable bytes')
        if len(self.codes) != self.rows * self.cols or len(self.scales) != self.rows * (self.cols // 32):
            raise ValueError('MXFP8 code or scale byte lengths disagree with geometry')
        _valid_bytes(self.codes, _BAD_E4M3, 'weights')
        _valid_bytes(self.scales, _BAD_UE8M0, 'scales')

    def at(self, row: int, col: int) -> float:
        if not 0 <= row < self.rows or not 0 <= col < self.cols:
            raise IndexError('MXFP8 matrix index out of bounds')
        return decode_e4m3fn(self.codes[row * self.cols + col]) * decode_ue8m0(
            self.scales[row * (self.cols // 32) + col // 32]
        )


def cpu_gemv(matrix: Matrix, x: Sequence[float]) -> list[float]:
    """Scalar, float64-accumulation correctness oracle for W[N,K] @ x[K]."""
    if len(x) != matrix.cols:
        raise ValueError('matvec input length disagrees with K')
    n_groups = matrix.cols // 32
    out = []
    for row in range(matrix.rows):
        terms = []
        for group in range(n_groups):
            scale = decode_ue8m0(matrix.scales[row * n_groups + group])
            index = row * matrix.cols + group * 32
            terms.extend(decode_e4m3fn(matrix.codes[index + k]) * scale * float(x[group * 32 + k])
                         for k in range(32))
        out.append(math.fsum(terms))
    return out


def torch_decode(matrix: Matrix, *, device: str = 'cpu'):
    """PyTorch reference DEQUANT on the selected device, not a native MXFP8 GEMM.

    The weight-byte reinterpretation is bit-preserving; never cast uint8 ->
    float8 numerically. Explicit exponent decoding avoids assuming E8M0 is FP32.
    """
    import torch

    raw_weights = torch.frombuffer(bytearray(matrix.codes), dtype=torch.uint8).reshape(matrix.rows, matrix.cols)
    raw_scales = torch.frombuffer(bytearray(matrix.scales), dtype=torch.uint8).reshape(matrix.rows, matrix.cols // 32)
    w = raw_weights.to(device=device).view(torch.float8_e4m3fn).to(torch.float32)
    scale_exponents = raw_scales.to(device=device).to(torch.int16) - 127
    power_of_two = torch.exp2(scale_exponents.to(torch.float32))
    return w * power_of_two.repeat_interleave(32, dim=1)


def torch_gemv(matrix: Matrix, x: Sequence[float], *, device: str = 'cpu'):
    """Materialize dequantized test weights, then use torch GEMV (CPU or CUDA)."""
    import torch

    if len(x) != matrix.cols:
        raise ValueError('matvec input length disagrees with K')
    inp = torch.as_tensor(x, dtype=torch.float32, device=device)
    return torch.matmul(torch_decode(matrix, device=device), inp)


def _window(rows: int, cols: int, first: int, count: int) -> None:
    if type(first) is not int or type(count) is not int or first < 0 or count < 1 or first + count > rows:
        raise ValueError('requested expert rows outside matrix geometry')
    if type(cols) is not int or cols < 32 or cols % 32:
        raise ValueError('MXFP8 source K must be divisible by 32')


def read_stage2a_rows(path: Path, *, expert_id: int, first: int, count: int) -> Matrix:
    """Read ONLY selected rows of one expert from an existing Stage-2A .i2bank."""
    from native_experts import FMT_MXFP8, read_bank_header

    path = Path(path)
    info = read_bank_header(path)
    if info['format_id'] != FMT_MXFP8 or info['layer'] < 20:
        raise ValueError('expected a last-four-layer MXFP8 Stage-2A expert bank')
    _window(info['rows'], info['cols'], first, count)
    if expert_id not in info['expert_ids']:
        raise ValueError('expert ID not present in staged bank')
    index = info['expert_ids'].index(expert_id)
    code_start = info['header_bytes'] + index * info['record_stride'] + first * info['cols']
    scale_start = (info['header_bytes'] + index * info['record_stride'] + info['code_stride']
                   + first * (info['cols'] // 32))
    with path.open('rb') as f:
        codes = os.pread(f.fileno(), count * info['cols'], code_start)
        scales = os.pread(f.fileno(), count * (info['cols'] // 32), scale_start)
    return Matrix(count, info['cols'], codes, scales)


def read_stage2b_rows(path: Path, *, layer: int, projection: str, expert_id: int,
                      first: int, count: int) -> Matrix:
    """Read selected original MXFP8 resource words from a NInfer-v3 Stage-2B carrier.

    On-demand reads. Never loads all banks or a full 14.6 GiB artifact.
    This is a reference-reader for the non-runtime-loadable carrier.
    """
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import ResourceObject
    from ninfer_artifact_bridge import LAYOUT

    if type(layer) is not int or not 20 <= layer < 24 or projection not in ('w13', 'w2'):
        raise ValueError('expected Inkling MXFP8 layer 20..23 and w13/w2 projection')
    with Artifact(path) as source:
        metadata = source.directory.metadata
        if metadata.get('storage_contract') != LAYOUT or metadata.get('runtime_loadable') is not False:
            raise ValueError('input is not a Stage-2B Inkling expert-only carrier')
        records = [r for r in metadata.get('inkling2_banks', [])
                   if r.get('layer') == layer and r.get('projection') == projection]
        if len(records) != 1:
            raise ValueError('missing or duplicate MXFP8 bank metadata')
        record = records[0]
        if (record.get('quantization') != 'mxfp8_e4m3_g32_ue8m0'
                or record.get('scale_group') != 32 or record.get('scale_dtype') != 'UE8M0_U8'):
            raise ValueError('MXFP8 carrier record has an unexpected format')
        ids = record['expert_ids']
        dims = record['shape']
        if (not isinstance(ids, list) or not ids or
                any(type(i) is not int or not 0 <= i < 128 for i in ids)
                or sorted(set(ids)) != ids or not isinstance(dims, list) or len(dims) != 3):
            raise ValueError('invalid MXFP8 carrier identity/geometry')
        n_experts, rows, cols = dims
        if type(n_experts) is not int or n_experts != len(ids) or type(rows) is not int:
            raise ValueError('invalid MXFP8 carrier shape')
        _window(rows, cols, first, count)
        if expert_id not in ids:
            raise ValueError('expert ID not present in carrier bank')
        prefix = f'inkling2/layer{layer:02d}/{projection}'
        if (record.get('codes') != prefix + '/mxfp8_codes'
                or record.get('scales') != prefix + '/mxfp8_ue8m0_scales'):
            raise ValueError('unexpected MXFP8 carrier resource IDs')
        id_obj = source.object(prefix + '/expert_ids')
        codes_obj = source.object(record['codes'])
        scales_obj = source.object(record['scales'])
        if not all(isinstance(obj, ResourceObject) and obj.encoding == 'raw_bytes_v1'
                   for obj in (id_obj, codes_obj, scales_obj)):
            raise ValueError('carrier MXFP8 words must be raw-byte resources')
        if (id_obj.bytes != 4 * n_experts or codes_obj.bytes != n_experts * rows * cols
                or scales_obj.bytes != n_experts * rows * (cols // 32)):
            raise ValueError('carrier MXFP8 resource size mismatch')
        if source.read_object(id_obj.id) != struct.pack(f'<{n_experts}I', *ids):
            raise ValueError('carrier expert-ID resource differs from metadata')
        index = ids.index(expert_id)
        codes = source.read_range(codes_obj.offset + index * rows * cols + first * cols, count * cols)
        scales = source.read_range(scales_obj.offset + index * rows * (cols // 32)
                                   + first * (cols // 32), count * (cols // 32))
        return Matrix(count, cols, codes, scales)


def synthetic_matrix(rows: int = 7, cols: int = 96) -> Matrix:
    """Deterministic nontrivial input with mixed signs/subnormals and K32 scales."""
    if rows < 1 or cols < 32 or cols % 32:
        raise ValueError('invalid synthetic MXFP8 shape')
    words = bytearray()
    for row in range(rows):
        for col in range(cols):
            word = (row * 73 + col * 19 + col // 32 * 29) & 255
            words.append(0x7e if (word & 0x7f) == 0x7f else word)
    codes = bytes(words)
    scales = bytes((118 + (row * 7 + block * 5) % 24)
                   for row in range(rows) for block in range(cols // 32))
    return Matrix(rows, cols, codes, scales)

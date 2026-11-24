"""Stage 5F: load precisely selected *real* Inkling 2 Mini weights from NInfer v3.

No torch dependency. Reads only the chosen expert planes (natural MXFP8, unswizzled
NVFP4) and router/embedding. No full-bank or full-embedding materialization.
This is a preparation bridge, NOT a full decoder or a ninfer-serve model loader.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Iterable

from ninfer_artifact_bridge import unswizzle_nvfp4_128_rows

HIDDEN = 2304
INTERMEDIATE = 1152
TOTAL_EXPERTS = 128
TOP_K = 8
KIND_NVFP4, KIND_MXFP8 = 1, 2


def bf16_to_fp32(raw: bytes) -> bytes:
    if len(raw) % 2:
        raise ValueError("BF16 bytes must have even length")
    # Exact, endian-independent conversion of BF16 words into float32 words.
    return b"".join(struct.pack("<I", word << 16)
                    for (word,) in struct.iter_unpack("<H", raw))


def fp32_values(raw: bytes) -> tuple[float, ...]:
    if len(raw) % 4:
        raise ValueError("FP32 bytes must have length divisible by 4")
    return tuple(v for (v,) in struct.iter_unpack("<f", raw))


def f32_bytes(values: Iterable[float]) -> bytes:
    result = array("f", (float(v) for v in values))
    if result.itemsize != 4:
        raise RuntimeError("requires IEEE-754 32-bit float")
    import sys
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def sigmoid(value: float) -> float:
    if value >= 0:
        e = math.exp(-value)
        return 1 / (1 + e)
    e = math.exp(value)
    return e / (1 + e)


def router_scores(hidden: tuple[float, ...], raw_w: bytes, raw_bias: bytes):
    if len(hidden) != HIDDEN or len(raw_w) != TOTAL_EXPERTS * HIDDEN * 4 or len(raw_bias) != 4 * TOTAL_EXPERTS:
        raise ValueError("unsupported real router geometry")
    weights = memoryview(raw_w).cast("f")
    bias = fp32_values(raw_bias)
    scores = []
    for e in range(TOTAL_EXPERTS):
        off = e * HIDDEN
        dot = math.fsum(hidden[i] * weights[off + i] for i in range(HIDDEN))
        score = sigmoid(dot)
        scores.append((score + bias[e], score, e))
    return sorted(scores, key=lambda t: (-t[0], t[2]))


@dataclass(frozen=True)
class RawExpert:
    codes: bytes
    scales: bytes
    amax: float = 1.0
    denominator: float = 1.0


class Carrier5F:
    """Read native NInfer-v3 carriers; construction validates their reported roles."""

    def __init__(self, nonexpert, experts):
        self.nonexpert = nonexpert
        self.experts = experts
        if nonexpert.directory.metadata.get("storage_contract") != "inkling2_hf_nonexpert_v3_v1":
            raise ValueError("expected Stage-4A non-expert carrier")
        if experts.directory.metadata.get("storage_contract") != "inkling2_ninfer_v3_expert_carrier_v1":
            raise ValueError("expected Stage-2B expert carrier")
        self.banks = {(b["layer"], b["projection"]): b
                      for b in experts.directory.metadata.get("inkling2_banks", [])}
        if len(self.banks) != len(experts.directory.metadata.get("inkling2_banks", [])):
            raise ValueError("duplicate expert bank identity")

    def _range(self, artifact, name: str, offset: int, length: int) -> bytes:
        obj = artifact.object(name)
        if offset < 0 or length <= 0 or offset + length > obj.bytes:
            raise ValueError(f"{name}: invalid object byte range")
        return artifact.read_range(obj.offset + offset, length)

    def nonexpert_tensor(self, name: str) -> tuple[bytes, str, tuple[int, ...]]:
        obj = self.nonexpert.object("hf/" + name)
        if obj.kind != "tensor" or obj.format not in ("fp32", "bf16"):
            raise ValueError(f"{name}: expected native BF16 or FP32 tensor")
        return self.nonexpert.read_object(obj.id), obj.format, tuple(obj.shape)

    def nonexpert_tensor_f32(self, name: str, expected_shape: tuple[int, ...]) -> bytes:
        raw, fmt, dims = self.nonexpert_tensor(name)
        if dims != expected_shape:
            raise ValueError(f"{name}: expected shape {expected_shape}, received {dims}")
        return bf16_to_fp32(raw) if fmt == "bf16" else raw

    def embedding_row(self, token_id: int) -> tuple[float, ...]:
        name = "hf/model.llm.embed.weight"
        obj = self.nonexpert.object(name)
        if obj.kind != "tensor" or obj.format != "bf16" or len(obj.shape) != 2 or obj.shape[1] != HIDDEN:
            raise ValueError("Inkling text embedding geometry or dtype unsupported")
        if type(token_id) is not int or not 0 <= token_id < obj.shape[0]:
            raise ValueError("token id is outside text embedding vocabulary")
        raw = self._range(self.nonexpert, name, token_id * HIDDEN * 2, HIDDEN * 2)
        return fp32_values(bf16_to_fp32(raw))

    def bank(self, layer: int, projection: str, ids: list[int]) -> list[RawExpert]:
        if layer not in range(24) or projection not in ("w13", "w2"):
            raise ValueError("invalid physical layer or expert projection")
        rec = self.banks.get((layer, projection))
        if rec is None or len(set(ids)) != len(ids):
            raise ValueError("bank missing or expert ids are repeated")
        all_ids = rec["expert_ids"]
        if rec["shape"] != [len(all_ids), HIDDEN, HIDDEN if projection == "w13" else INTERMEDIATE]:
            raise ValueError("unexpected expert bank geometry")
        index = {id_: pos for pos, id_ in enumerate(all_ids)}
        if any(type(e) is not int or e not in index for e in ids):
            raise ValueError("requested expert missing from carrier")
        prefix = f"inkling2/layer{layer:02d}/{projection}"
        n, k = HIDDEN, HIDDEN if projection == "w13" else INTERMEDIATE
        kind = KIND_NVFP4 if layer < 20 else KIND_MXFP8
        result = []
        if kind == KIND_NVFP4:
            if rec.get("quantization") != "nvfp4_e2m1_g16_e4m3":
                raise ValueError("wrong NVFP4 bank contract")
            object_id = prefix + "/nvfp4"
            obj = self.experts.object(object_id)
            if obj.kind != "tensor" or obj.layout != "batched_block_scale_k16_m128x4_v1" or obj.format != "nvfp4":
                raise ValueError("NVFP4 native layout mismatch")
            # Rows 0..N are packed, scales have 128-row tiled NInfer swizzle.
            stride_codes = n * k // 2
            stride_scales = n * k // 16
            plane_scale_offset = (len(all_ids) * stride_codes + 255) // 256 * 256
            for e in ids:
                pos = index[e]
                codes = self._range(self.experts, object_id, pos * stride_codes, stride_codes)
                swizzled = self._range(self.experts, object_id,
                                       plane_scale_offset + pos * stride_scales, stride_scales)
                tile_stride = 128 * k // 16
                scales = b"".join(unswizzle_nvfp4_128_rows(swizzled[i:i + tile_stride], k)
                                   for i in range(0, len(swizzled), tile_stride))
                scalar_bytes = self._range(self.experts, prefix + "/original_amax_denominator_fp32", pos * 8, 8)
                amax, den = struct.unpack("<ff", scalar_bytes)
                if not (math.isfinite(amax) and math.isfinite(den) and amax > 0 and den > 0):
                    raise ValueError("invalid original NVFP4 amax/denominator")
                result.append(RawExpert(codes, scales, amax, den))
        else:
            if (rec.get("quantization") != "mxfp8_e4m3_g32_ue8m0" or
                    rec.get("scale_group") != 32):
                raise ValueError("wrong MXFP8 bank contract")
            stride_codes = n * k
            stride_scales = n * k // 32
            for e in ids:
                pos = index[e]
                codes = self._range(self.experts, prefix + "/mxfp8_codes", pos * stride_codes, stride_codes)
                scales = self._range(self.experts, prefix + "/mxfp8_ue8m0_scales", pos * stride_scales, stride_scales)
                result.append(RawExpert(codes, scales))
        return result


def validate_complete_carriers(checkpoint_root: Path, nonexpert, experts, bank_dir: Path) -> dict:
    """Full metadata and origin validation, WITHOUT reading all 18GB of weight bytes."""
    from nonexpert_artifact import Checkpoint
    from runtime_bindings import create_binding_plan
    with Checkpoint(checkpoint_root) as checkpoint:
        plan = create_binding_plan(checkpoint, nonexpert=nonexpert,
                                   experts=experts, banks_dir=bank_dir)
        if not plan["coverage"]["metadata_complete"]:
            raise ValueError("full model weight carriers/bank provenance are incomplete")
    return plan

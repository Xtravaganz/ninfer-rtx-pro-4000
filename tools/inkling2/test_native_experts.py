"""Stdlib-only regression tests for lossless Inkling2 expert-bank staging."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

from native_experts import (
    FMT_MXFP8, FMT_NVFP4, HEADER, IndexedSafetensors, MiniGeometry,
    decode_fp8, decode_mxfp8_row, decode_nvfp4_row, describe_expert_bank,
    export_bank, format_for_layer, parse_selection, read_bank_header, verify_bank,
)


SMALL = MiniGeometry(experts=2, rows=8, gate_up_cols=64, down_cols=32)


def fake_safetensors(folder: Path, overrides: dict[str, bytes] | None = None) -> dict[str, bytes]:
    """Generate tiny NVFP4/MXFP8 banks with real layer names and physical layouts."""
    tensors = {}
    for layer in (0, 20):
        for proj, k in (("w13", SMALL.gate_up_cols), ("w2", SMALL.down_cols)):
            stem = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
            if layer == 0:
                # Each expert has unique low/high nibbles. Natural row-major scales.
                tensors[stem] = ("U8", (2, SMALL.rows, k//2),
                                 bytes([0x31, 0x42]) * (SMALL.experts*SMALL.rows*k//4))
                tensors[stem + "_scale"] = ("F8_E4M3", (2, SMALL.rows, k//16),
                                            bytes([0x38]) * (2*SMALL.rows*k//16))
                tensors[stem + "_amax"] = ("F32", (2,), struct.pack("<ff", 2.0, 4.0))
                tensors[stem + "_outer_scale_denominator"] = ("F32", (1,), struct.pack("<f", 8.0))
            else:
                tensors[stem] = ("F8_E4M3", (2, SMALL.rows, k),
                                 bytes([0x38, 0xB8]) * (SMALL.experts*SMALL.rows*k//2))
                tensors[stem + "_scale"] = ("U8", (2, SMALL.rows, k//32),
                                            bytes([127]) * (2*SMALL.rows*k//32))
    overrides = overrides or {}
    for name, contents in overrides.items():
        dtype, shape, _ = tensors[name]
        tensors[name] = (dtype, shape, contents)
    header = {}
    blob = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        expected = (4 if dtype == "F32" else 1)
        for s in shape:
            expected *= s
        if len(raw) != expected:
            raise ValueError(f"invalid synthetic bytes for {name}: {len(raw)} != {expected}")
        begin = len(blob)
        blob.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [begin, len(blob)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    shard = "model-00001-of-00001.safetensors"
    (folder / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    (folder / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for name in tensors}
    }))
    return {name: raw for name, (_, _, raw) in tensors.items()}


class TestBankEncoding(unittest.TestCase):
    def test_config_and_selector(self):
        self.assertEqual(parse_selection("0,3-5,5", 24), (0,3,4,5))
        self.assertEqual(parse_selection("all", 2), (0,1))
        for value in ("", "-1", "24", "0,", "4-1", "1-3-5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_selection(value, 24)
        self.assertEqual([format_for_layer(x) for x in (0,19,20,23)], [1,1,2,2])
        self.assertEqual(HEADER.size, 64)

    def test_numeric_reference(self):
        self.assertEqual(decode_fp8(0x38), 1)
        self.assertEqual(decode_fp8(0xB8), -1)
        self.assertEqual(decode_fp8(0x30), 0.5)
        self.assertEqual(decode_fp8(0x01), 2**-9)
        with self.assertRaises(ValueError):
            decode_fp8(0x7F)
        # 0x21 means [0.5, 1.0] with E4M3 scale 1; multiply by amax/den 0.25.
        self.assertEqual(decode_nvfp4_row(b"\x21" * 8, b"\x38", 2, 8),
                         [0.125, 0.25] * 8)
        self.assertEqual(decode_mxfp8_row(bytes([0x38, 0xB8])*16, bytes([128])),
                         [2.0, -2.0]*16)
        for fn, args in ((decode_mxfp8_row, (bytes([0x38])*32, b"\xff")),
                         (decode_nvfp4_row, (b"\x21"*8, b"\x7f", 1, 1))):
            with self.assertRaises(ValueError):
                fn(*args)

    def test_native_bank_roundtrip_and_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            originals = fake_safetensors(folder)
            output = folder.parent / (folder.name + "-output")
            try:
                with IndexedSafetensors(folder) as store:
                    for layer in (0, 20):
                        for projection in ("w13", "w2"):
                            bank = describe_expert_bank(store, layer, projection, SMALL)
                            path = output / f"layer{layer}_{projection}.i2bank"
                            entry = export_bank(store, bank, (0, 1), path, chunk_bytes=7)
                            hdr = read_bank_header(path)
                            self.assertEqual((hdr["layer"], hdr["projection"], hdr["expert_ids"]),
                                             (layer, projection, (0, 1)))
                            self.assertEqual(entry["sha256"], verify_bank(
                                store, bank, path, entry["sha256"], chunk_bytes=11)["sha256"])
                            self.assertEqual(path.stat().st_size, 64 + 8 + 2 * bank.record_stride)
                            stem = bank.name()
                            body = path.read_bytes()[72:]
                            for expert in (0,1):
                                record = body[expert*bank.record_stride:(expert+1)*bank.record_stride]
                                start = expert * bank.code_stride
                                self.assertEqual(record[:bank.code_stride], originals[stem][start:start+bank.code_stride])
                                sstart = expert * bank.scale_stride
                                self.assertEqual(record[bank.code_stride:bank.code_stride+bank.scale_stride],
                                                 originals[stem+"_scale"][sstart:sstart+bank.scale_stride])
                            with self.assertRaises(FileExistsError):
                                export_bank(store, bank, (0,1), path)
                            # Tampering must be caught by byte equality, not only file SHA.
                            corrupted = bytearray(path.read_bytes())
                            corrupted[-1] ^= 1
                            path.write_bytes(corrupted)
                            with self.assertRaises(ValueError):
                                verify_bank(store, bank, path)
            finally:
                import shutil
                shutil.rmtree(output, ignore_errors=True)

    def test_bad_dtype_and_quant_words(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            stem = "model.llm.layers.20.mlp.experts.w13_weight_scale"
            # All 1-byte scales, last one reserved 255.
            bad = bytes([127]) * (2*SMALL.rows*SMALL.gate_up_cols//32 - 1) + b"\xff"
            fake_safetensors(folder, {stem: bad})
            output = folder / "data.bin"
            with IndexedSafetensors(folder) as store:
                bank = describe_expert_bank(store, 20, "w13", SMALL)
                with self.assertRaisesRegex(ValueError, "invalid quantization word"):
                    export_bank(store, bank, (1,), output, chunk_bytes=5)
                self.assertFalse(output.exists())
                self.assertFalse(list(folder.glob("data.bin.*.part")))

    def test_invalid_source_shape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            fake_safetensors(folder)
            with IndexedSafetensors(folder) as store:
                with self.assertRaisesRegex(ValueError, "expected U8"):
                    describe_expert_bank(store, 0, "w13", MiniGeometry())
                with self.assertRaises(ValueError):
                    describe_expert_bank(store, 24, "w13", SMALL)


if __name__ == "__main__":
    unittest.main()

class TestMetadataOnlyPlan(unittest.TestCase):
    def test_plan_uses_stage1_config_without_weights(self):
        from check_checkpoint import expected_parameters
        from export_experts import main
        from contextlib import redirect_stdout
        import io
        root = Path(__file__).resolve().parents[2]
        config = root / 'tests/models/inkling2/fixtures/config.json'
        if not config.is_file():
            self.skipTest('Stage1 fixture missing; run after installing Stage1')
        names = list(expected_parameters())
        names += ['model.llm.middle_cycle_cross_pass_queries',
                  'model.llm.middle_cycle_cross_pass_kv_queries',
                  'model.llm.embed.weight', 'model.llm.unembed.weight']
        names += [f'model.llm.attn_res.mlp_queries.{i}' for i in range(24)]
        with tempfile.TemporaryDirectory() as d:
            index = Path(d) / 'index.json'
            index.write_text(json.dumps({'weight_map': {name: 'not-downloaded.safetensors' for name in names}}))
            out = io.StringIO()
            with redirect_stdout(out):
                result = main(['--plan', '--config', str(config), '--index', str(index),
                               '--layers', '0,20', '--experts', '0-1'])
            self.assertEqual(result, 0)
            self.assertIn('PLAN ONLY', out.getvalue())
            self.assertIn('2 layers', out.getvalue())

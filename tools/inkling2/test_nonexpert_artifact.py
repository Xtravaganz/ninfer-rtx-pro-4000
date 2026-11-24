"""Stage4A contract, security, and actual NInfer-v3 integration tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from nonexpert_artifact import (
    Checkpoint, DTYPE_BYTES, STORAGE_CONTRACT, _span_from_header,
    describe_plan, is_expert_source, pack_remaining, payload_estimate,
    verify_remaining,
)

HERE = Path(__file__).resolve().parent


def fixture(root: Path, entries: list[tuple[str, str, tuple[int, ...], bytes]],
            *, name: str = "model-00001-of-00001.safetensors") -> None:
    """Make a *tiny* valid safetensors main shard; no 19GB fixture needed."""
    config_path = HERE.parents[1] / "tests/models/inkling2/fixtures/config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
    else:
        # Tests for metadata preflight run outside a full NInfer checkout too.
        config = {"architectures": ["Inkling2LoopForConditionalGeneration"],
                  "text_config": {"model_type": "inkling2_loop_text"},
                  "vision_config": {"model_type": "inkling2_vision"},
                  "eos_token_id": 200006}
    (root / "config.json").write_text(json.dumps(config))
    header = {"__metadata__": {"format": "pt"}}
    flat = bytearray()
    for tensor, dtype, shape, data in entries:
        assert len(data) == (1 if not shape else __import__("math").prod(shape)) * DTYPE_BYTES[dtype]
        header[tensor] = {"dtype": dtype, "shape": list(shape),
                          "data_offsets": [len(flat), len(flat) + len(data)]}
        flat.extend(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    (root / name).write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + flat)
    index = {"metadata": {"total_size": len(flat)},
             "weight_map": {tensor: name for tensor, *_ in entries}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))


class TestMetadata(unittest.TestCase):
    def test_expert_source_is_precise(self):
        self.assertTrue(is_expert_source("model.llm.layers.0.mlp.experts.w13_weight"))
        self.assertTrue(is_expert_source("model.llm.layers.19.mlp.experts.w2_weight_amax"))
        self.assertTrue(is_expert_source("model.llm.layers.20.mlp.experts.w2_weight_scale"))
        self.assertFalse(is_expert_source("model.llm.layers.0.mlp.gate.weight"))
        with self.assertRaisesRegex(ValueError, "unknown Inkling expert"):
            is_expert_source("model.llm.layers.0.mlp.experts.strange.weight")
        with self.assertRaisesRegex(ValueError, "unexpectedly"):
            is_expert_source("model.llm.layers.20.mlp.experts.w2_weight_amax")

    def test_plan_never_reads_weight_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [
                ("model.llm.layers.0.mlp.gate.weight", "BF16", (2, 2), bytes(range(8))),
                ("model.llm.layers.0.mlp.gate.bias", "F32", (2,), struct.pack("<ff", 0.4, 1.5)),
                ("model.vision.blocks.0.weight", "F16", (3,), bytes(range(6))),
            ])
            with Checkpoint(root) as c:
                def forbid(*args):
                    raise AssertionError("unexpected tensor read during plan")
                c.read = forbid
                self.assertEqual(len(c.remaining_names), 3)
                self.assertEqual(describe_plan(c.select())["formats"],
                                 {"bf16": 1, "fp32": 1, "int32": 0, "raw": 1})
                self.assertEqual([s.dtype for s in c.select(("model.llm.layers.0.mlp.gate.*",))],
                                 ["F32", "BF16"])
                with self.assertRaisesRegex(ValueError, "matches no"):
                    c.select(("misspelled.*",))
                self.assertEqual(c.index_sha256, __import__("hashlib").sha256(
                    (root / "model.safetensors.index.json").read_bytes()).hexdigest())

    def test_partition_keeps_router_vision_and_loop_but_not_experts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [
                ("model.llm.layers.0.mlp.experts.w13_weight", "U8", (2,), b"\x00\x01"),
                ("model.llm.layers.0.mlp.gate.weight", "BF16", (2,), b"\x01\x02\x03\x04"),
                ("model.llm.middle_cycle_cross_pass_queries", "F32", (1,), b"\x00\x00\x80\x3f"),
                ("model.vision.encoder.layer.0.weight", "BF16", (1,), b"\x00\x01"),
            ])
            with Checkpoint(root) as c:
                self.assertEqual(len(c.expert_names), 1)
                self.assertEqual(len(c.remaining_names), 3)
                self.assertIn("model.llm.layers.0.mlp.gate.weight", c.remaining_names)
                self.assertIn("model.llm.middle_cycle_cross_pass_queries", c.remaining_names)
                self.assertIn("model.vision.encoder.layer.0.weight", c.remaining_names)

    def test_multiple_shards_and_index_independence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first, second = "model-00001.safetensors", "model-00002.safetensors"
            fixture(root, [("a", "BF16", (1,), b"\x10\x20")], name=first)
            fixture(root, [("b", "F32", (1,), struct.pack("<f", 3.25))], name=second)
            idx = root / "model.safetensors.index.json"
            data = json.loads(idx.read_text())
            data["weight_map"]["a"] = first
            idx.write_text(json.dumps(data))
            with Checkpoint(root) as c:
                self.assertEqual(len(c.spans), 2)
                self.assertEqual(c.read(c.spans["a"], 0, 2), b"\x10\x20")
                self.assertEqual(c.read(c.spans["b"], 0, 4), struct.pack("<f", 3.25))

    def test_duplicate_json_members_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [("a", "BF16", (1,), b"\x10\x20")])
            idx = root / "model.safetensors.index.json"
            idx.write_text('{"weight_map":{"a":"model-00001-of-00001.safetensors","a":"model-00001-of-00001.safetensors"}}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                Checkpoint(root)

    def test_alignment_size_estimator(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [
                ("a", "U8", (3,), b"abc"),
                ("b", "BF16", (1,), b"\x00\x00"),
                ("c", "F16", (1,), b"\x01\x02"),
                ("d", "F32", (1,), b"\x00\x00\x80\x3f"),
            ])
            with Checkpoint(root) as c:
                # a:3, b:aligned 256 +2, c:260, d:aligned 512 +4
                self.assertEqual(payload_estimate(c.select()), 516)

    def test_corrupt_index_and_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [("a", "BF16", (1,), b"\x10\x20")])
            idx = root / "model.safetensors.index.json"
            data = json.loads(idx.read_text())
            data["weight_map"]["invented"] = "model-00001-of-00001.safetensors"
            idx.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "header/index mismatch"):
                Checkpoint(root)
            del data["weight_map"]["invented"]
            idx.write_text(json.dumps(data))
            with Checkpoint(root) as c:
                span = c.spans["a"]
                self.assertEqual(c.read(span, 0, 2), b"\x10\x20")
                with self.assertRaisesRegex(ValueError, "out-of-bounds"):
                    c.read(span, 1, 2)
            (root / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", 99999999))
            with self.assertRaisesRegex(ValueError, "header size"):
                Checkpoint(root)

    def test_corrupt_geometry_dtype_overlap(self):
        path = Path("/tmp/example.safetensors")
        for record in (
            {"dtype": "BF16", "shape": [3], "data_offsets": [0, 5]},
            {"dtype": "Q9", "shape": [1], "data_offsets": [0, 1]},
            {"dtype": "BF16", "shape": [0], "data_offsets": [0, 0]},
            {"dtype": "BF16", "shape": [1], "data_offsets": [-1, 1]},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                _span_from_header("bad", record, path, 100, 1000)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [("a", "U8", (2,), b"ab"), ("b", "U8", (2,), b"cd")])
            shard = root / "model-00001-of-00001.safetensors"
            binary = shard.read_bytes()
            header_len = struct.unpack("<Q", binary[:8])[0]
            header = json.loads(binary[8:8 + header_len])
            header["b"]["data_offsets"] = [1, 3]  # still width 2, but overlaps a
            new = json.dumps(header, separators=(",", ":")).encode()
            shard.write_bytes(struct.pack("<Q", len(new)) + new + binary[8 + header_len:])
            with self.assertRaisesRegex(ValueError, "overlapping tensor"):
                Checkpoint(root)

    def test_main_model_index_excludes_mtp_and_refuses_incomplete_strict(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, [("model.llm.embed.weight", "BF16", (1,), b"\x00\x01")])
            (root / "mtp.safetensors").write_bytes(b"unrelated file")
            with Checkpoint(root) as c:
                self.assertEqual(c.remaining_names, ("model.llm.embed.weight",))
                with self.assertRaises((ValueError, KeyError)):
                    c.validate_inkling()

    def test_cli_help_direct_python_without_pythonpath(self):
        script = HERE / "pack_nonexpert.py"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, str(script), "--help"],
                                cwd="/tmp", env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--verify", result.stdout)


class TestNativeV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.artifact.reader import Artifact  # noqa: F401
            from tools.artifact.writer import ArtifactWriter  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("NInfer v3 repository modules unavailable")

    def test_roundtrip_bf16_fp32_int32_and_opaque_f16(self):
        from tools.artifact.reader import Artifact
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "src"
            root.mkdir()
            fixture(root, [
                ("model.llm.layers.0.mlp.gate.weight", "BF16", (2, 2), bytes(range(8))),
                ("model.llm.layers.0.mlp.gate.bias", "F32", (2,), struct.pack("<ff", 0.5, 1.5)),
                ("model.llm.middle_cycle_cross_pass_queries", "I32", (2,), struct.pack("<ii", 3, -4)),
                ("model.vision.some_tensor", "F16", (4,), bytes(range(8, 16))),
            ])
            dest = Path(d) / "remainder.ninfer"
            with Checkpoint(root) as c:
                with self.assertRaisesRegex(ValueError, "require-complete"):
                    pack_remaining(c, dest, includes=("model.llm.layers.0.*",),
                                   require_complete=True)
                result = pack_remaining(c, dest, chunk_bytes=3,
                                        max_file_bytes=1 << 20)
                self.assertEqual(result["tensors"], 4)
                self.assertEqual(verify_remaining(c, dest, require_complete=True,
                                                  chunk_bytes=2)["tensors"], 4)
                with Artifact(dest) as artifact:
                    self.assertFalse(artifact.directory.metadata["runtime_loadable"])
                    self.assertEqual(artifact.directory.metadata["storage_contract"], STORAGE_CONTRACT)
                    self.assertEqual(len(artifact.directory.bindings), 3)
                    self.assertEqual(len(artifact.objects), 4)
                with self.assertRaises(FileExistsError):
                    pack_remaining(c, dest)

    def test_subset_and_data_corruption_detected(self):
        from tools.artifact.reader import Artifact
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "src"
            root.mkdir()
            fixture(root, [("a", "BF16", (2,), b"\x01\x02\x03\x04"),
                           ("b", "F16", (4,), b"\x04\x05\x06\x07\x08\x09\x0a\x0b")])
            dest = Path(d) / "subset.ninfer"
            with Checkpoint(root) as c:
                pack_remaining(c, dest, includes=("a",), chunk_bytes=1)
                self.assertEqual(verify_remaining(c, dest, includes=("a",))["tensors"], 1)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    verify_remaining(c, dest, require_complete=True)
                with self.assertRaisesRegex(ValueError, "selection"):
                    verify_remaining(c, dest, includes=("b",))
                with Artifact(dest) as art:
                    obj = art.object("hf/a")
                    byte_offset = art.payload_offset + obj.offset
                with dest.open("r+b") as f:
                    f.seek(byte_offset)
                    f.write(b"\xff")
                with self.assertRaisesRegex(ValueError, "byte mismatch"):
                    verify_remaining(c, dest)


if __name__ == "__main__":
    unittest.main()

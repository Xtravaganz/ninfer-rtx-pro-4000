"""Stage 4C: official checkpoint loop-controller and full-attention norm roles.

This suite intentionally does not need the 20 GB checkpoint or torch.  A separate
on-machine `bind_model.py --plan` checks the actual HF index and tensor headers.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from runtime_bindings import classify

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def controller_name(layer: int, stream: str, part: str) -> str:
    return f"model.llm.layers.{layer}.controllers.{stream}.{part}_weight"


def controller_shape(part: str) -> tuple[int, int]:
    return (64, 4608) if part == "down" else (2304, 64)


class TestOfficialLoopRoles(unittest.TestCase):
    def test_all_144_official_controller_slots(self) -> None:
        roles = []
        for layer in range(6, 18):
            for stream in ("attn", "k", "v", "mlp"):
                for part in ("down", "forget", "gain"):
                    name = controller_name(layer, stream, part)
                    role = classify(name, "BF16", controller_shape(part))
                    self.assertEqual((role.category, role.layer), ("loop_control", layer))
                    self.assertEqual(role.source, name)
                    self.assertEqual(role.role, f"text.layer{layer:02d}.controllers.{stream}.{part}_weight")
                    self.assertEqual(role.shape, controller_shape(part))
                    roles.append(role.role)
        self.assertEqual(len(roles), 144)
        self.assertEqual(len(set(roles)), 144)

    def test_all_12_full_attention_input_norms(self) -> None:
        names = []
        for layer in (5, 11, 17, 23):
            for stream in ("q", "k", "v"):
                name = f"model.llm.layers.{layer}.{stream}_input_norm.weight"
                role = classify(name, "F32", (2304,))
                self.assertEqual((role.category, role.layer), ("normalization", layer))
                self.assertEqual(role.role, f"text.layer{layer:02d}.{stream}_input_norm.weight")
                names.append(name)
        self.assertEqual(len(set(names)), 12)

    def test_controller_wrong_layers_are_rejected(self) -> None:
        for layer in (0, 5, 18, 23):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "outside physical core"):
                classify(controller_name(layer, "attn", "down"), "BF16", (64, 4608))

    def test_controller_dtype_and_shape_are_rejected(self) -> None:
        for dtype, shape in (("F32", (64, 4608)), ("BF16", (4608, 64)),
                             ("BF16", (64, 2304)), ("U8", (64, 4608))):
            with self.subTest(dtype=dtype, shape=shape), self.assertRaisesRegex(ValueError, "expected BF16"):
                classify(controller_name(7, "k", "down"), dtype, shape)
        with self.assertRaisesRegex(ValueError, "expected BF16"):
            classify(controller_name(17, "mlp", "gain"), "BF16", (64, 2304))

    def test_full_norm_wrong_layer_and_geometry_are_rejected(self) -> None:
        for layer in (0, 6, 10, 18):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "sliding layer"):
                classify(f"model.llm.layers.{layer}.q_input_norm.weight", "F32", (2304,))
        for dtype, shape in (("I32", (2304,)), ("F32", (1024,)), ("BF16", (2304, 1))):
            with self.subTest(dtype=dtype, shape=shape), self.assertRaisesRegex(ValueError, "expected FP32/BF16"):
                classify("model.llm.layers.11.v_input_norm.weight", dtype, shape)

    def test_unknown_revision_does_not_silently_classify(self) -> None:
        unknown = ("model.llm.layers.7.controllers.attn.alien_weight",
                   "model.llm.layers.7.controllers.foo.down_weight",
                   "model.llm.layers.11.z_input_norm.weight")
        for name in unknown:
            with self.subTest(name=name):
                self.assertEqual(classify(name, "BF16", (2304,)).category, "unresolved")

    def test_stage4b_roles_remain_stable(self) -> None:
        cases = (
            ("model.llm.layers.7.mlp.gate.weight", "router"),
            ("model.llm.layers.7.attn.q_proj.weight", "attention"),
            ("model.llm.layers.7.attn_sconv.weight", "short_convolution"),
            ("model.llm.layers.7.attn_norm.weight", "normalization"),
            ("model.visual.encoder.0.weight", "vision"),
        )
        for name, category in cases:
            with self.subTest(name=name):
                self.assertEqual(classify(name, "BF16", (2304,)).category, category)

    def test_packer_cli_direct_and_module_mode_without_pythonpath(self) -> None:
        script = HERE / "pack_ninfer_artifact.py"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        commands = (
            [sys.executable, str(script), "--help"],
            [sys.executable, "-m", "tools.inkling2.pack_ninfer_artifact", "--help"],
        )
        for args in commands:
            with self.subTest(args=args):
                result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--require-complete", result.stdout)


if __name__ == "__main__":
    unittest.main()

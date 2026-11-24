"""Standalone tests; run: python3 -m unittest tools/inkling2/test_contract.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

script = Path(__file__).with_name('check_checkpoint.py')
spec = importlib.util.spec_from_file_location('inkling2_checkpoint', script)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
fixture = Path(__file__).resolve().parents[2] / 'tests/models/inkling2/fixtures/config.json'


class TestContract(unittest.TestCase):
    def test_config(self):
        config = json.loads(fixture.read_text())
        module.validate_config(config)
        config['text_config']['loop_config']['num_passes'] = 2
        with self.assertRaises(ValueError):
            module.validate_config(config)

    def test_header(self):
        expected = module.expected_parameters()
        header = {key: {'dtype': dtype, 'shape': shape} for key, (dtype, shape) in expected.items()}
        module.validate_header(header)
        header['model.llm.layers.20.mlp.experts.w13_weight_scale']['dtype'] = 'BF16'
        with self.assertRaises(ValueError):
            module.validate_header(header)

    def test_index(self):
        names = dict.fromkeys(module.expected_parameters(), 'model.safetensors')
        names.update({
            'model.llm.embed.weight': 'model.safetensors',
            'model.llm.unembed.weight': 'model.safetensors',
            'model.llm.middle_cycle_cross_pass_queries': 'model.safetensors',
            'model.llm.middle_cycle_cross_pass_kv_queries': 'model.safetensors',
        })
        names.update({f'model.llm.attn_res.mlp_queries.{i}': 'model.safetensors'
                      for i in range(24)})
        module.validate_index({'weight_map': names})
        names.pop('model.llm.layers.0.mlp.gate.weight')
        with self.assertRaises(ValueError):
            module.validate_index({'weight_map': names})


if __name__ == '__main__':
    unittest.main()

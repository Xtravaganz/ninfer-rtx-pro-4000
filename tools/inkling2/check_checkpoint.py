#!/usr/bin/env python3
"""Offline Inkling 2 Mini tensor-contract checks (no weights, no torch).

Usage:
  python3 tools/inkling2/check_checkpoint.py --config config.json --index model.safetensors.index.json
  python3 tools/inkling2/check_checkpoint.py --config config.json --index ... --shard model.safetensors
  python3 tools/inkling2/check_checkpoint.py --config config.json --index ... --header header.json

`--shard` reads only the first 8 + header_length bytes. No tensor payload is read.
`--header` accepts an extracted Safetensors JSON header.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(c: dict) -> None:
    require(c.get('architectures') == ['Inkling2LoopForConditionalGeneration'], 'architecture mismatch')
    require(c.get('model_type') == 'inkling2_loop', 'model_type mismatch')
    t = c['text_config']
    expected = dict(model_type='inkling2_loop_text', hidden_size=2304, num_hidden_layers=24,
                    vocab_size=201024, unpadded_vocab_size=200058, n_routed_experts=128,
                    num_experts_per_tok=8, intermediate_size=1152, sliding_window_size=1024,
                    n_layers_per_attn_res_block=6)
    for key, value in expected.items():
        require(t.get(key) == value, f'text_config.{key}: expected {value!r}, got {t.get(key)!r}')
    require(t.get('loop_config') == dict(core_start=6, core_end=18, num_passes=3,
                                         gate_rank=64, is_varloop=False), 'loop configuration mismatch')
    require(t.get('local_layer_ids') == [i for i in range(24) if i % 6 != 5],
            'local/full attention profile mismatch')
    require(c.get('quantization_config', {}).get('quant_algo') == 'NVFP4', 'quantization mismatch')


def expected_parameters() -> dict[str, tuple[str, list[int]]]:
    tensors: dict[str, tuple[str, list[int]]] = {}
    for i in range(24):
        prefix = f'model.llm.layers.{i}.mlp.'
        tensors[prefix + 'gate.weight'] = ('BF16', [128, 2304])
        tensors[prefix + 'gate.bias'] = ('F32', [128])
        for proj, k in [('w13', 2304), ('w2', 1152)]:
            base = prefix + f'experts.{proj}_weight'
            if i < 20:
                tensors[base] = ('U8', [128, 2304, k // 2])
                tensors[base + '_scale'] = ('F8_E4M3', [128, 2304, k // 16])
                tensors[base + '_amax'] = ('F32', [128])
                tensors[base + '_outer_scale_denominator'] = ('F32', [1])
            else:
                tensors[base] = ('F8_E4M3', [128, 2304, k])
                tensors[base + '_scale'] = ('U8', [128, 2304, k // 32])
    return tensors


def validate_index(index: dict) -> None:
    weight_map = index['weight_map']
    require(isinstance(weight_map, dict), 'invalid safetensors weight_map')
    required = set(expected_parameters()) | {
        'model.llm.middle_cycle_cross_pass_queries',
        'model.llm.middle_cycle_cross_pass_kv_queries',
        'model.llm.embed.weight',
        'model.llm.unembed.weight',
    }
    for i in range(24):
        required.add(f'model.llm.attn_res.mlp_queries.{i}')
    missing = sorted(required.difference(weight_map))
    require(not missing, f'missing {len(missing)} essential checkpoint tensors: {missing[:8]}')


def read_shard_header(path: Path) -> dict:
    with path.open('rb') as f:
        length_bytes = f.read(8)
        require(len(length_bytes) == 8, 'truncated safetensors prefix')
        size, = struct.unpack('<Q', length_bytes)
        require(0 < size <= 128 * 1024 * 1024, f'unsafe/unexpected header length: {size}')
        raw = f.read(size)
        require(len(raw) == size, 'truncated safetensors header')
    return json.loads(raw)


def validate_header(header: dict) -> None:
    fields = expected_parameters()
    for name, (dtype, shape) in fields.items():
        require(name in header, f'missing header field {name}')
        item = header[name]
        require(item['dtype'] == dtype and item['shape'] == shape,
                f'{name}: expected {dtype} {shape}, got {item["dtype"]} {item["shape"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--header', type=Path)
    source.add_argument('--shard', type=Path)
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    index = json.loads(a.index.read_text())
    validate_config(config)
    validate_index(index)
    print(f'OK: Inkling 2 Mini HF config; {len(index["weight_map"])} indexed tensor names')
    if a.header or a.shard:
        header = json.loads(a.header.read_text()) if a.header else read_shard_header(a.shard)
        validate_header(header)
        print(f'OK: {len(expected_parameters())} router/expert tensor dtypes and shapes; '
              'NVFP4 layers 0..19, MXFP8 layers 20..23')
    else:
        print('NOTE: dtype/shape verification requires --shard or --header')


if __name__ == '__main__':
    main()

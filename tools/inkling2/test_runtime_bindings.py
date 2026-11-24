"""Stage 4B: logical role selection, loop schedule, cross-carrier validation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from nonexpert_artifact import Checkpoint, pack_remaining, STORAGE_CONTRACT
from ninfer_artifact_bridge import LAYOUT as EXPERT_CONTRACT
from runtime_bindings import (CONTRACT, classify, classify_all, schedule,
                              inspect_nonexpert, inspect_experts,
                              create_binding_plan, write_plan)
from test_nonexpert_artifact import fixture

HERE = Path(__file__).resolve().parent
CFG = HERE.parents[1] / 'tests/models/inkling2/fixtures/config.json'


def config():
    return json.loads(CFG.read_text())


def obj(id, kind='resource', bytes=1, shape=None, fmt=None, layout=None):
    return NS(id=id, kind=kind, bytes=bytes, shape=shape, format=fmt,
              layout=layout, encoding='raw_bytes_v1')


class FakeV3:
    def __init__(self, metadata, provenance, objects, bindings=None, resources=None, cfg=None,
                 ids_bytes=None):
        self.path = Path('/tmp/synthetic-4b.ninfer')
        self.artifact_id = b'0' * 16
        self.objects = tuple(objects)
        self.by_id = {x.id: x for x in objects}
        cfg = config() if cfg is None else cfg
        text = {**cfg['text_config'], 'model_type': 'inkling2_loop_text',
                'architectures': cfg['architectures'], 'eos_token_id': cfg['eos_token_id']}
        self.directory = NS(metadata=metadata, provenance=provenance,
                            components={'text': {'config': text, 'resources': resources or {}}},
                            bindings=bindings or {})
        self.ids_bytes = ids_bytes or {}

    def object(self, name):
        return self.by_id[name]

    def read_object(self, name):
        return self.ids_bytes[name]


def nonexpert_fake(c, names):
    records = [c.spans[n].record() for n in names]
    objects = [obj('hf/'+n, 'tensor' if c.spans[n].direct_format else 'resource',
                   c.spans[n].byte_count, c.spans[n].shape, c.spans[n].direct_format, 'contiguous_le_v1')
               for n in names]
    bindings = {n: {'object': 'hf/'+n} for n in names if c.spans[n].direct_format}
    res = {'hf/'+n: 'hf/'+n for n in names if not c.spans[n].direct_format}
    return FakeV3({'storage_contract': STORAGE_CONTRACT, 'runtime_loadable': False,
                   'main_checkpoint_only': True, 'tensors': records},
                  {'model_index_sha256': c.index_sha256}, objects, bindings, res, c.config)


def expert_fake(c, layers=(0, 20)):
    records = []
    objects = []
    bindings = {}
    resource_ids = {}
    for layer in layers:
        proj, ids = 'w13', [0, 3]
        prefix = f'inkling2/layer{layer:02d}/{proj}'
        id_name = prefix + '/expert_ids'
        resource_ids[id_name] = struct.pack('<II', 0, 3)
        objects.append(obj(id_name, bytes=8))
        if layer < 20:
            t, scal = prefix + '/nvfp4', prefix + '/original_amax_denominator_fp32'
            objects.append(obj(t, 'tensor', 1, (2, 2304, 2304), 'nvfp4',
                               'batched_block_scale_k16_m128x4_v1'))
            objects.append(obj(scal, bytes=16))
            bindings[t] = {'object': t}
            r = dict(layer=layer, projection=proj, expert_ids=ids, shape=[2,2304,2304],
                     quantization='nvfp4_e2m1_g16_e4m3', native_tensor=t,
                     original_scalars=scal, divisor_policy='fp32(outer_denominator / amax)')
        else:
            codes, scales = prefix + '/mxfp8_codes', prefix + '/mxfp8_ue8m0_scales'
            objects.append(obj(codes, bytes=2*2304*2304))
            objects.append(obj(scales, bytes=2*2304*2304//32))
            r = dict(layer=layer, projection=proj, expert_ids=ids, shape=[2,2304,2304],
                     quantization='mxfp8_e4m3_g32_ue8m0', codes=codes, scales=scales,
                     scale_group=32, scale_dtype='UE8M0_U8')
        records.append(r)
    resources = {x.id: x.id for x in objects if x.kind == 'resource'}
    return FakeV3({'storage_contract': EXPERT_CONTRACT, 'runtime_loadable': False,
                   'inkling2_banks': records}, {}, objects, bindings, resources, c.config,
                  ids_bytes=resource_ids)


class TestRoles(unittest.TestCase):
    def test_known_roles_and_original_names(self):
        cases = {
            'model.llm.embed.weight': ('embedding', None),
            'model.llm.unembed.weight': ('lm_head', None),
            'model.llm.attn_res.mlp_queries.23': ('block_residual', 23),
            'model.llm.middle_cycle_cross_pass_queries': ('loop_control', None),
            'model.llm.layers.7.mlp.gate.weight': ('router', 7),
            'model.llm.layers.7.attn.q_proj.weight': ('attention', 7),
            'model.llm.layers.7.attn_sconv.weight': ('short_convolution', 7),
            'model.llm.layers.7.mlp_norm.weight': ('normalization', 7),
            'model.llm.layers.7.q_message_gate.down_proj.weight': ('loop_control', 7),
            'model.visual.encoder.0.weight': ('vision', None),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                role = classify(name, 'BF16', (2304,))
                self.assertEqual((role.category, role.layer), expected)
                self.assertEqual(role.source, name)
                self.assertEqual(role.shape, (2304,))

    def test_unknown_revisions_do_not_silently_become_supported(self):
        self.assertEqual(classify('model.audio.foo.weight','BF16',(1,)).category, 'unresolved')
        self.assertEqual(classify('model.llm.layers.5.q_message_gate.weight','BF16',(1,)).category, 'unresolved')
        with self.assertRaisesRegex(ValueError, 'outside'):
            classify('model.llm.layers.24.attn.k.weight', 'BF16', (1,))
        with self.assertRaisesRegex(ValueError, 'outside'):
            classify('model.llm.attn_res.mlp_queries.24', 'BF16', (1,))
        with self.assertRaisesRegex(ValueError, 'expert source'):
            classify('model.llm.layers.0.mlp.experts.w13_weight', 'U8', (1,))

    def test_48_visit_plan_matches_stage1_slots_and_resets(self):
        s = schedule(config())
        self.assertEqual(len(s), 48)
        self.assertEqual(sorted(x['cache_slot'] for x in s), list(range(48)))
        self.assertEqual([x['physical_layer'] for x in s[:6]], list(range(6)))
        self.assertEqual([x['physical_layer'] for x in s[-6:]], list(range(18,24)))
        self.assertEqual([x['ordinal'] for x in s if x['reset_to_prelude']], [18,30])
        self.assertEqual([x['cache_slot'] for x in s if x['physical_layer'] == 6], [6,24,36])
        self.assertEqual(len([x for x in s if x['reads_previous_pass_messages']]), 24)
        self.assertEqual([x['attention'] for x in s if x['physical_layer'] == 17], ['full']*3)
        bad = config()
        bad['text_config']['loop_config']['num_passes'] = 2
        with self.assertRaisesRegex(ValueError, 'schedule'):
            schedule(bad)


class TestCarriers(unittest.TestCase):
    def make_checkpoint(self, path):
        fixture(path, [
            ('model.llm.layers.0.mlp.gate.weight', 'BF16', (2,2), b'abcdefgh'),
            ('model.llm.layers.0.mlp.gate.bias', 'F32', (2,), struct.pack('<ff', 0.5,1.5)),
            ('model.llm.embed.weight', 'BF16', (2,), b'1234'),
            ('model.visual.some.weight', 'F16', (2,), b'\x01\x02\x03\x04'),
        ])
        return Checkpoint(path)

    def test_partial_nonexpert_binding_and_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.make_checkpoint(Path(d)) as c:
                a = nonexpert_fake(c, c.remaining_names[:2])
                rows, complete = inspect_nonexpert(a, c)
                self.assertFalse(complete)
                self.assertEqual(len(rows), 2)
                a.directory.provenance['model_index_sha256'] = 'bad'
                with self.assertRaisesRegex(ValueError, 'different HF index'):
                    inspect_nonexpert(a, c)

    def test_full_nonexpert_binding_and_metadata_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            with self.make_checkpoint(Path(d)) as c:
                a = nonexpert_fake(c, c.remaining_names)
                rows, complete = inspect_nonexpert(a, c)
                self.assertTrue(complete)
                self.assertEqual(len(rows), 4)
                a.directory.metadata['tensors'][0]['dtype'] = 'F32'
                with self.assertRaisesRegex(ValueError, 'does not match HF'):
                    inspect_nonexpert(a, c)

    def test_expert_bank_id_bytes_and_unexpected_precision_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.make_checkpoint(Path(d)) as c:
                a = expert_fake(c)
                r, full, origin = inspect_experts(a, c)
                self.assertEqual(len(r), 2)
                self.assertFalse(full)
                self.assertFalse(origin)
                k = 'inkling2/layer00/w13/expert_ids'
                a.ids_bytes[k] = b'\x00'*8
                with self.assertRaisesRegex(ValueError, 'ID bytes'):
                    inspect_experts(a,c)
                a.ids_bytes[k] = struct.pack('<II',0,3)
                a.directory.metadata['inkling2_banks'][1]['scale_group'] = 64
                with self.assertRaisesRegex(ValueError, 'MXFP8 bank format'):
                    inspect_experts(a,c)
                a.directory.metadata['inkling2_banks'][1]['scale_group'] = 32
                del a.directory.components['text']['resources'][k]
                with self.assertRaisesRegex(ValueError, 'resource map'):
                    inspect_experts(a,c)

    def test_expert_origin_rejects_unrelated_hf_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'src'
            source.mkdir()
            with self.make_checkpoint(source) as c:
                a = expert_fake(c)
                banks = root / 'banks'
                banks.mkdir()
                (banks / 'manifest.json').write_text(json.dumps({
                    'source_index_sha256': 'f'*64,
                    'source_config_sha256': '0'*64,
                    'layout': 'inkling2_native_expert_bank_v1', 'version': 1,
                    'banks': []}))
                with self.assertRaisesRegex(ValueError, 'different HF snapshot'):
                    inspect_experts(a, c, banks_dir=banks)

    def test_loop_binding_uses_same_physical_carriers_on_three_passes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.make_checkpoint(Path(d)) as c:
                c.validate_inkling = lambda: None  # tiny fixture, not a full HF checkpoint
                a, b = nonexpert_fake(c, c.remaining_names), expert_fake(c)
                report = create_binding_plan(c, nonexpert=a, experts=b)
                self.assertEqual(report['contract'], CONTRACT)
                self.assertFalse(report['runtime_loadable'])
                self.assertFalse(report['coverage']['metadata_complete'])
                self.assertEqual(report['coverage']['expert_banks_bound'], 2)
                slots = [s for s in report['execution']['steps'] if s['physical_layer']==6]
                self.assertEqual([s['cache_slot'] for s in slots], [6,24,36])
                self.assertEqual({tuple(s['expert_bank_ids']) for s in slots},
                                 {('inkling2/layer06/w13','inkling2/layer06/w2')})

    def test_manifest_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'roles.json'
            write_plan(path, {'x': 1})
            self.assertEqual(json.loads(path.read_text()), {'x': 1})
            with self.assertRaises(FileExistsError):
                write_plan(path, {'x': 2})

    def test_direct_cli_help_without_pythonpath(self):
        p = HERE / 'bind_model.py'
        env = dict(__import__('os').environ)
        env.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable,str(p),'--help'],
                                cwd='/tmp', env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--require-complete', result.stdout)


class TestRealArtifact(unittest.TestCase):
    def test_real_ninfer_writer_reader_nonexpert(self):
        try:
            from tools.artifact.reader import Artifact
            from tools.artifact.writer import ArtifactWriter
        except ImportError:
            self.skipTest('requires full NInfer repository for ArtifactWriter/Artifact')
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            fixture(src, [
                ('model.llm.embed.weight', 'BF16', (2,2), b'abcdefgh'),
                ('model.llm.layers.0.mlp.gate.bias', 'F32', (2,), struct.pack('<ff',0.1,0.2)),
            ])
            with Checkpoint(src) as c:
                dest = Path(d)/'remaining.ninfer'
                pack_remaining(c, dest)
                with Artifact(dest) as a:
                    rows, complete = inspect_nonexpert(a,c)
                    self.assertTrue(complete)
                    self.assertEqual(len(rows),2)


if __name__ == '__main__':
    unittest.main()

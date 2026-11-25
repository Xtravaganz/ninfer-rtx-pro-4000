"""Stage 5L+5M: metadata and original-weight-style loop mathematics contracts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from run_real_core_stage5lm import core_source_shapes,core_preflight

try:
    import torch
    from real_sliding_stage5h import Geometry,TorchCarrierWeights
    from real_core_stage5lm import (
        _cross_queries,_controller,core_one_token_layer,run_real_core,
    )
    from real_full_stage5j import run_first_attention_block
    from test_real_sliding_stage5hi import FakeCarrier
    from test_real_block_stage5jk import TestIntegratedBlock
except ImportError:
    torch=None


def official():
    file=HERE.parents[1]/'tests/models/inkling2/fixtures/config.json'
    return json.loads(file.read_text()) if file.exists() else None


class TestContracts(unittest.TestCase):
    def test_layer_metadata_source_shapes(self):
        cfg=official()
        if cfg is None:self.skipTest('requires official config fixture')
        a=core_source_shapes(6,cfg)
        self.assertEqual(a['model.llm.attn_res.attn_queries.5'],((2304,),))
        self.assertEqual(a['model.llm.layers.6.attn.k_sconv.weight'],((1024,4),(1024,1,4)))
        self.assertNotIn('model.llm.layers.6.attn.k_sconv.static_w',a)
        b=core_source_shapes(11,cfg)
        self.assertEqual(b['model.llm.attn_res.attn_queries.10'],((3,2304),))
        self.assertEqual(b['model.llm.layers.11.attn.k_sconv.static_w'],((4,1024),))
        self.assertEqual(b['model.llm.layers.11.attn.k_sconv.weight_proj'],((32,2304),))
        c=core_source_shapes(17,cfg)
        self.assertEqual(c['model.llm.attn_res.attn_queries.16'],((3,2304),))
        for bad in (5,18,True,11.0):
            with self.assertRaises(ValueError):core_source_shapes(bad,cfg)

    def test_metadata_completeness_and_caches(self):
        cfg=official()
        if cfg is None:self.skipTest('requires official config fixture')
        from run_real_sliding_stage5hi import mandatory_sliding_sources
        from run_real_block_stage5jk import mandatory_full_sources
        t=cfg['text_config'];d=t['hidden_size']
        req={'model.llm.embed.weight':((t['vocab_size'],d),),
             'model.llm.embed_norm.weight':((d,),)}
        for i in range(5):req.update(mandatory_sliding_sources(i,cfg))
        req.update(mandatory_full_sources(cfg))
        for i in range(6,18):req.update(core_source_shapes(i,cfg))
        for i in range(6,18):
            for channel in ('attn','k','v','mlp'):
                p=f'model.llm.layers.{i}.controllers.{channel}.'
                req[p+'down_weight']=((64,4608),)
                req[p+'forget_weight']=((2304,64),)
                req[p+'gain_weight']=((2304,64),)
        req['model.llm.middle_cycle_cross_pass_queries']=((24,d),)
        req['model.llm.middle_cycle_cross_pass_kv_queries']=((24,d),)
        spans={name:SimpleNamespace(dtype='BF16' if '.controllers.' in name else 'F32',shape=opt[0]) for name,opt in req.items()}
        p6=core_preflight(spans,cfg,core_layers=6,passes=1)
        self.assertEqual(p6['planned_visits'],12)
        self.assertEqual(p6['cache_slots'],list(range(12)))
        p12=core_preflight(spans,cfg,core_layers=12,passes=1)
        self.assertEqual(p12['planned_visits'],18)
        self.assertEqual(p12['controller_tensors'],0)
        p36=core_preflight(spans,cfg,core_layers=12,passes=3)
        self.assertEqual(p36['planned_visits'],42)
        self.assertEqual(p36['controller_tensors'],144)
        self.assertEqual(p36['cache_slots'][:18],list(range(18)))
        self.assertEqual(p36['cache_slots'][18:30],list(range(24,36)))
        self.assertEqual(p36['cache_slots'][30:42],list(range(36,48)))
        self.assertFalse(p36['has_postlude'])
        missing=dict(spans);del missing['model.llm.layers.11.attn.k_sconv.static_w']
        with self.assertRaisesRegex(ValueError,'unsupported HF tensor contract'):
            core_preflight(missing,cfg,core_layers=12,passes=1)
        bad=dict(spans);bad['model.llm.layers.6.controllers.attn.down_weight']=SimpleNamespace(dtype='F32',shape=(1,1))
        with self.assertRaisesRegex(ValueError,'unsupported HF tensor contract'):
            core_preflight(bad,cfg,core_layers=12,passes=3)
        for i,k in ((6,2),(0,1),(12,0),(6,0),(True,1),(6,1.0)):
            with self.assertRaises(ValueError):core_preflight(spans,cfg,core_layers=i,passes=k)


@unittest.skipIf(torch is None,'requires torch')
class TestCoreNumerics(unittest.TestCase):
    @staticmethod
    def make_weights():
        torch.manual_seed(7)
        torch.set_num_threads(1)
        g=Geometry(hidden=32,intermediate=16,query_heads=4,kv_heads=2,
                   head_dim=8,rel_dim=4,rel_extent=8,experts=16,top_k=8,conv_kernel=4)
        carrier=FakeCarrier()
        # Stage 5J full-attention canonical compact weights for layer 5.
        full=TestIntegratedBlock.make()
        for name,x in full.reader.params.items():
            if name.startswith('model.llm.layers.5.') or name in (
                'model.llm.attn_res.attn_queries.4','model.llm.attn_res.mlp_queries.5'):
                carrier.params[name]=x.detach().clone()
        source=dict(carrier.params)
        for layer in range(6,18):
            full_layer=layer in (11,17)
            parent=5 if full_layer else 1
            layer_prefix=f'model.llm.layers.{parent}.'
            query=f'model.llm.attn_res.attn_queries.{parent-1}'
            mlp=f'model.llm.attn_res.mlp_queries.{parent}'
            for name,x in source.items():
                if name.startswith(layer_prefix):
                    carrier.params[name.replace(layer_prefix,f'model.llm.layers.{layer}.',1)]=x.detach().clone()
                elif name==query:
                    carrier.params[f'model.llm.attn_res.attn_queries.{layer-1}']=x.detach().clone()
                elif name==mlp:
                    carrier.params[f'model.llm.attn_res.mlp_queries.{layer}']=x.detach().clone()
            for path in ('attn','k','v','mlp'):
                pref=f'model.llm.layers.{layer}.controllers.{path}.'
                carrier.params[pref+'down_weight']=torch.zeros((4,64))
                carrier.params[pref+'forget_weight']=torch.zeros((32,4))
                carrier.params[pref+'gain_weight']=torch.zeros((32,4))
        qm=torch.arange(24*32,dtype=torch.float32).reshape(24,32)*.0001
        kv=(1000+torch.arange(24*32,dtype=torch.float32)).reshape(24,32)*.0001
        carrier.params['model.llm.middle_cycle_cross_pass_queries']=qm
        carrier.params['model.llm.middle_cycle_cross_pass_kv_queries']=kv
        return TorchCarrierWeights(carrier,'cpu',g)

    def test_query_order_uses_official_interleave(self):
        w=self.make_weights()
        actual=_cross_queries(w)
        qm=w.reader.params['model.llm.middle_cycle_cross_pass_queries'].reshape(12,2,32)
        kv=w.reader.params['model.llm.middle_cycle_cross_pass_kv_queries'].reshape(12,2,32)
        for i in range(12):
            self.assertTrue(torch.equal(actual[i,0],qm[i,0]))
            self.assertTrue(torch.equal(actual[i,1],kv[i,0]))
            self.assertTrue(torch.equal(actual[i,2],kv[i,1]))
            self.assertTrue(torch.equal(actual[i,3],qm[i,1]))

    def test_core_first_boundary_appends_incoming_hidden_and_reset(self):
        w=self.make_weights()
        hidden,blocks,_=run_first_attention_block(w,9,6)
        before=hidden.clone()
        output,blocks2,rec=core_one_token_layer(w,hidden,blocks,layer=6,gate_rank=4)
        self.assertEqual(tuple(blocks2.shape),(1,1,2,32))
        self.assertTrue(torch.equal(blocks2[:,:,1,:],before))
        self.assertEqual(rec['physical_layer'],6)
        self.assertEqual(rec['block_states'],2)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_first_half_has_full_attention_11(self):
        w=self.make_weights()
        hidden,blocks,trace=run_real_core(w,9,core_layers=6,passes=1,gate_rank=4)
        self.assertEqual(len(trace),12)
        self.assertEqual([r['physical_layer'] for r in trace],list(range(12)))
        self.assertEqual(trace[11]['attention_type'],'full')
        self.assertEqual(int(blocks.shape[-2]),2)
        self.assertTrue(torch.isfinite(hidden).all())

    def test_full_first_core_has_two_block_boundaries(self):
        w=self.make_weights()
        hidden,blocks,trace=run_real_core(w,9,core_layers=12,passes=1,gate_rank=4)
        self.assertEqual(len(trace),18)
        self.assertEqual(trace[11]['attention_type'],'full')
        self.assertEqual(trace[17]['attention_type'],'full')
        self.assertEqual(int(blocks.shape[-2]),3)
        self.assertTrue(torch.isfinite(hidden).all())

    def test_three_passes_reset_and_have_disjoint_slots(self):
        a=self.make_weights();b=self.make_weights()
        first,first_blocks,first_trace=run_real_core(a,9,core_layers=12,passes=1,gate_rank=4)
        last,blocks,trace=run_real_core(b,9,core_layers=12,passes=3,gate_rank=4)
        # All compact controller weights are zero: the second and third
        # passes MUST restart with the same prelude hidden and block states.
        self.assertTrue(torch.equal(first,last))
        self.assertTrue(torch.equal(first_blocks,blocks))
        self.assertEqual(len(trace),42)
        self.assertEqual([r['core_pass'] for r in trace[6:]], [0]*12+[1]*12+[2]*12)
        self.assertEqual([r['cache_slot'] for r in trace[6:18]],list(range(6,18)))
        self.assertEqual([r['cache_slot'] for r in trace[18:30]],list(range(24,36)))
        self.assertEqual([r['cache_slot'] for r in trace[30:42]],list(range(36,48)))
        self.assertTrue(all(r['messages_applied'] for r in trace[18:]))
        self.assertFalse(any(r['messages_applied'] for r in trace[6:18]))

    def test_nonzero_controller_changes_second_pass(self):
        baseline=self.make_weights()
        modified=self.make_weights()
        p='model.llm.layers.6.controllers.attn.'
        modified.reader.params[p+'down_weight'][0,:]=.25
        modified.reader.params[p+'gain_weight'][:,0]=.15
        # Test the controller directly to avoid the deliberately saturated,
        # constant FakeCarrier expert weights masking an upstream difference.
        hidden=torch.linspace(-2,2,32).reshape(1,1,32)
        message=torch.linspace(1,-3,32).reshape(1,1,32)
        a=_controller(baseline,6,'attn',hidden,message,4)
        b=_controller(modified,6,'attn',hidden,message,4)
        self.assertTrue(torch.equal(a,hidden))
        self.assertGreater(float((a-b).abs().max()),1e-7)

    def test_malformed_messages_rejected(self):
        w=self.make_weights()
        hidden,blocks,_=run_first_attention_block(w,9,6)
        for layer in (5,18,True,11.0):
            with self.assertRaises(ValueError):core_one_token_layer(w,hidden,blocks,layer=layer)
        for msg in (torch.zeros(1,1,4,32),torch.zeros(1,1,3,32)):
            with self.assertRaises(ValueError):core_one_token_layer(w,hidden,blocks,layer=6,message=msg,core_pass=0)
        with self.assertRaises(ValueError):core_one_token_layer(w,hidden,blocks,layer=6,core_pass=1)
        with self.assertRaises(ValueError):run_real_core(w,9,core_layers=6,passes=2,gate_rank=4)


if __name__=='__main__':unittest.main()

"""Stage 5J/5K self-contained geometry, mathematics, and integration regressions.

The real checkpoint GPU smoke is intentionally separate (run_real_block_stage5jk.sh).
Passing these tests does NOT establish HF or full-model parity.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))

from run_real_block_stage5jk import mandatory_full_sources, first_block_preflight
try:
    import torch
    from real_sliding_stage5h import Geometry, TorchCarrierWeights, run_prelude
    from real_full_stage5j import full_attention_one_token_layer, run_first_attention_block
    from test_real_sliding_stage5hi import FakeCarrier
except ImportError:
    torch=None


def official_cfg():
    fixture=HERE.parents[1]/'tests/models/inkling2/fixtures/config.json'
    if not fixture.exists():
        return None
    return json.loads(fixture.read_text())


class TestMetadata(unittest.TestCase):
    def test_full_layer_uses_three_attn_queries_and_no_sliding_convs(self):
        cfg=official_cfg()
        if cfg is None: self.skipTest('requires Stage1 HF config fixture')
        sh=mandatory_full_sources(cfg)
        self.assertEqual(sh['model.llm.attn_res.attn_queries.4'],((3,2304),))
        self.assertEqual(sh['model.llm.attn_res.mlp_queries.5'],((2304,),))
        self.assertEqual(sh['model.llm.layers.5.attn.k_sconv.static_w'],((4,1024),))
        self.assertEqual(sh['model.llm.layers.5.attn.v_sconv.weight_proj'],((32,2304),))
        self.assertIn('model.llm.layers.5.q_input_norm.weight',sh)
        self.assertIn('model.llm.layers.5.k_input_norm.weight',sh)
        self.assertIn('model.llm.layers.5.v_input_norm.weight',sh)
        self.assertNotIn('model.llm.layers.5.attn.k_sconv.weight',sh)
        self.assertNotIn('model.llm.attn_res.attn_queries.5',sh)
        self.assertEqual(len(sh),24)

    def test_modified_architecture_fails_closed(self):
        cfg=official_cfg()
        if cfg is None: self.skipTest('requires Stage1 HF config fixture')
        for case in (
            lambda x: x['text_config'].update(rel_extent=9),
            lambda x: x['text_config']['local_layer_ids'].append(5),
            lambda x: x['text_config'].update(num_attention_heads=16),
        ):
            changed=json.loads(json.dumps(cfg))
            case(changed)
            with self.assertRaises(ValueError):mandatory_full_sources(changed)

    def test_metadata_preflight_geometry_and_offset(self):
        cfg=official_cfg()
        if cfg is None: self.skipTest('requires Stage1 HF config fixture')
        from run_real_sliding_stage5hi import mandatory_sliding_sources
        d=cfg['text_config']['hidden_size']
        shapes={
            'model.llm.embed.weight':((cfg['text_config']['vocab_size'],d),),
            'model.llm.embed_norm.weight':((d,),),
        }
        for l in range(5):shapes.update(mandatory_sliding_sources(l,cfg))
        shapes.update(mandatory_full_sources(cfg))
        spans={name:SimpleNamespace(dtype='BF16',shape=sh[0]) for name,sh in shapes.items()}
        good=first_block_preflight(spans,cfg,layers=6)
        self.assertEqual(good['executed_physical_layers'],[0,1,2,3,4,5])
        self.assertEqual(good['first_visit_cache_slots'],[0,1,2,3,4,5])
        self.assertTrue(good['has_dynamic_kv_convolution'])
        self.assertFalse(good['full_forward_supported'])
        good5=first_block_preflight(spans,cfg,layers=5)
        self.assertFalse(good5['has_dynamic_kv_convolution'])
        spans2=dict(spans);del spans2['model.llm.layers.5.attn.k_sconv.static_w']
        with self.assertRaisesRegex(ValueError,'unsupported full-layer weights'):
            first_block_preflight(spans2,cfg,layers=6)
        # This is the exact offset failure which broke the original Stage 5I.
        spans2=dict(spans);spans2['model.llm.attn_res.attn_queries.4']=SimpleNamespace(dtype='F32',shape=(d,))
        with self.assertRaisesRegex(ValueError,'unsupported full-layer weights'):
            first_block_preflight(spans2,cfg,layers=6)
        spans2=dict(spans);spans2['model.llm.layers.5.attn.v_sconv.weight_proj']=SimpleNamespace(dtype='U8',shape=(32,d))
        with self.assertRaisesRegex(ValueError,'unsupported full-layer weights'):
            first_block_preflight(spans2,cfg,layers=6)
        for value in (0,4,7,48,True,6.0):
            with self.assertRaises(ValueError):first_block_preflight(spans,cfg,layers=value)


@unittest.skipIf(torch is None,'requires torch')
class TestIntegratedBlock(unittest.TestCase):
    @staticmethod
    def make(device='cpu'):
        g=Geometry(hidden=32,intermediate=16,query_heads=4,kv_heads=2,
                   head_dim=8,rel_dim=4,rel_extent=8,experts=16,
                   top_k=8,conv_kernel=4)
        carrier=FakeCarrier()
        prefix='model.llm.layers.5.';a=prefix+'attn.'
        d=32;hq=4;hk=2;k=4;rel=4
        shapes={
            prefix+'q_input_norm.weight':(d,), prefix+'k_input_norm.weight':(d,),
            prefix+'v_input_norm.weight':(d,), prefix+'mlp_norm.weight':(d,),
            prefix+'attn_sconv.weight':(d,k),prefix+'mlp_sconv.weight':(d,k),
            prefix+'mlp.gate.weight':(16,d),prefix+'mlp.gate.bias':(16,),
            a+'wq_du.weight':(hq*8,d),a+'wk_dv.weight':(hk*8,d),
            a+'wv_dv.weight':(hk*8,d),a+'wo_ud.weight':(d,hq*8),
            a+'gate_attn.weight':(hq*8,d),a+'attn_sink_proj.weight':(hq,d),
            a+'q_l2_clip.scale':(hq,8),a+'rel_l2_clip.scale':(hq,rel),
            a+'rel_logits_proj.proj':(rel,8),a+'wr_du.weight':(hq*rel,d),
            a+'k_sconv.static_w':(k,hk*8),a+'v_sconv.static_w':(k,hk*8),
            a+'k_sconv.weight_proj':(hk*k,d),a+'v_sconv.weight_proj':(hk*k,d),
            'model.llm.attn_res.attn_queries.4':(3,d),
            'model.llm.attn_res.mlp_queries.5':(d,),
        }
        torch.manual_seed(1234)
        for name,shape in shapes.items():
            tensor=torch.randn(shape,dtype=torch.float32)*0.006
            if name.endswith('norm.weight') or name.endswith('clip.scale'):
                tensor=torch.ones(shape,dtype=torch.float32)
            if name.endswith('gate.bias'):
                tensor=torch.arange(16,dtype=torch.float32)*0.003
            carrier.params[name]=tensor
        return TorchCarrierWeights(carrier,device,g)

    def test_full_layer_realstyle_mathematics_and_block_ownership(self):
        w=self.make()
        hidden,blocks,_=run_prelude(w,9,5)
        before=blocks.clone()
        out,blocks,record=full_attention_one_token_layer(w,hidden,blocks)
        self.assertEqual(tuple(out.shape),(1,1,32))
        self.assertEqual(tuple(blocks.shape),(1,1,1,32))
        self.assertTrue(torch.equal(blocks,before))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(record['dynamic_kv_convolution'])
        self.assertEqual(record['physical_layer'],5)
        self.assertEqual(len(record['top8_expert_ids']),8)
        self.assertTrue(any(layer==5 for layer,_,_ in w.reader.locs))
        self.assertGreater(float((out-hidden).abs().max()),0.)

    def test_baseline_five_reproduces_old_stage5i(self):
        w=self.make();a,ba,ta=run_prelude(w,17,5)
        w2=self.make();b,bb,tb=run_first_attention_block(w2,17,5)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(ba,bb))
        self.assertEqual([r['physical_layer'] for r in tb],list(range(5)))

    def test_six_layer_vs_explicit_composition(self):
        w=self.make();first,blocks,_=run_prelude(w,17,5)
        final_a,blocks_a,_=full_attention_one_token_layer(w,first,blocks)
        w2=self.make();final_b,blocks_b,trace=run_first_attention_block(w2,17,6)
        self.assertTrue(torch.allclose(final_a,final_b,rtol=1e-5,atol=1e-6))
        self.assertTrue(torch.equal(blocks_a,blocks_b))
        self.assertEqual([x['physical_layer'] for x in trace],list(range(6)))
        self.assertTrue(torch.isfinite(final_b).all())

    def test_dynamic_kv_projection_uses_per_token_coefficients(self):
        # Stage5J uses this exact Stage5A primitive for the original full K/V.
        from forward_oracle_stage5a import dynamic_short_convolution
        projection=torch.zeros((8,32),dtype=torch.float32)
        zeros=torch.zeros((4,16),dtype=torch.float32)
        x=torch.ones((1,1,16),dtype=torch.float32)
        dynamic_input=torch.ones((1,1,32),dtype=torch.float32)
        base=dynamic_short_convolution(x,dynamic_input,zeros,projection,num_heads=2)
        projection[0,:]=0.125  # changes current-token tap for KV-head 0
        changed=dynamic_short_convolution(x,dynamic_input,zeros,projection,num_heads=2)
        self.assertGreater(float((base-changed).abs().max()),0.1)
        # The other head has no dynamic perturbation.
        self.assertTrue(torch.equal(base[...,8:],changed[...,8:]))

    def test_wrong_position_and_unimplemented_layers_rejected(self):
        w=self.make();h,blocks,_=run_prelude(w,10,5)
        for layer in (0,4,6,17,23,True):
            with self.assertRaises(ValueError):
                full_attention_one_token_layer(w,h,blocks,layer=layer)
        for position in (1,-1,True,1.2):
            with self.assertRaises(ValueError):
                full_attention_one_token_layer(w,h,blocks,position=position)
        with self.assertRaises(ValueError):
            full_attention_one_token_layer(w,h,torch.empty(1,1,0,32))
        for count in (0,4,7,48):
            with self.assertRaises(ValueError):
                run_first_attention_block(self.make(),7,count)

    def test_shape_mismatch_fails_closed(self):
        w=self.make();h,blocks,_=run_prelude(w,10,5)
        bad='model.llm.layers.5.attn.k_sconv.static_w'
        w.reader.params[bad]=torch.zeros((1,16))
        with self.assertRaises(ValueError):
            full_attention_one_token_layer(w,h,blocks)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'requires CUDA')
    def test_cuda_matches_cpu_one_full_layer(self):
        cpu=self.make('cpu');gpu=self.make('cuda:0')
        a,_,_=run_first_attention_block(cpu,100,6)
        b,_,_=run_first_attention_block(gpu,100,6)
        self.assertTrue(torch.allclose(a,b.cpu(),atol=2e-3,rtol=2e-3))


if __name__=='__main__':unittest.main()

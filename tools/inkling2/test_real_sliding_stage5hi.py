"""Stage 5H/5I deterministic compact analogues; real checkpoint runs are separate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
try:
    import torch
    from real_sliding_stage5h import Geometry,TorchCarrierWeights,decode_expert,run_prelude,sliding_one_token_layer
except ImportError:
    torch=None
from run_real_sliding_stage5hi import mandatory_sliding_sources, readiness_from_spans


class Source(SimpleNamespace):
    pass


def config():
    return {'text_config':{'vocab_size':1234, 'hidden_size':32,'head_dim':8,
        'num_attention_heads':4,'num_key_value_heads':2,'d_rel':4,
        'sliding_window_size':8,'sconv_kernel_size':4,'n_routed_experts':16}}


class FakeCarrier:
    def __init__(self,device='cpu'):
        self.params={}
        g=config()['text_config']; D=g['hidden_size']; Hq=4;hk=2;d=8;rel=4;K=4
        self.params['model.llm.embed_norm.weight']=torch.ones(D)
        self.locs=[]
        torch.manual_seed(3)
        for i in range(5):
            p=f'model.llm.layers.{i}.';a=p+'attn.'
            sh={p+'attn_norm.weight':(D,),p+'mlp_norm.weight':(D,),
                p+'attn_sconv.weight':(D,K),p+'mlp_sconv.weight':(D,K),
                p+'mlp.gate.weight':(16,D),p+'mlp.gate.bias':(16,),
                a+'wq_du.weight':(Hq*d,D),a+'wk_dv.weight':(hk*d,D),
                a+'wv_dv.weight':(hk*d,D),a+'wo_ud.weight':(D,Hq*d),
                a+'gate_attn.weight':(Hq*d,D), a+'attn_sink_proj.weight':(Hq,D),
                a+'k_sconv.weight':(hk*d,K),a+'v_sconv.weight':(hk*d,K),
                a+'q_l2_clip.scale':(Hq,d),a+'rel_l2_clip.scale':(Hq,rel),
                a+'rel_logits_proj.proj':(rel,8), a+'wr_du.weight':(Hq*rel,D),
                f'model.llm.attn_res.mlp_queries.{i}':(D,)}
            if i:
                sh[f'model.llm.attn_res.attn_queries.{i-1}']=(D,)
            for name,shape in sh.items():
                self.params[name] = torch.randn(shape)*.006
                if name.endswith('norm.weight') or name.endswith('clip.scale'):
                    self.params[name]=torch.ones(shape)
            self.params[p+'mlp.gate.bias']=torch.arange(16).float()*0.003
        self.embedding_data=tuple((j%9 - 4)*.02 for j in range(D))

    def embedding_row(self,token):
        return self.embedding_data

    def nonexpert_tensor(self,name):
        v=self.params[name]
        return v.contiguous().numpy().tobytes(),'fp32',tuple(v.shape)

    def bank(self,layer,projection,ids):
        self.locs.append((layer,projection,tuple(ids)))
        D=32;I=16
        n,k=(2*I,D) if projection=='w13' else (D,I)
        # Constant exactly-representable NVFP4=1 scale=1 and E2M1 nibble=2.
        return [Source(codes=bytes([0x22])*(n*k//2),
                       scales=bytes([0x38])*(n*k//16),amax=1.,denominator=1.) for _ in ids]


class TestMetadata(unittest.TestCase):
    def test_reject_pretend_full_48(self):
        for bad in (-1,5,6,48,5.5,True):
            with self.assertRaises(ValueError):
                mandatory_sliding_sources(bad,config())
    def test_required_names_and_geometry(self):
        c=config();names=mandatory_sliding_sources(4,c)
        self.assertIn('model.llm.attn_res.attn_queries.3',names)
        self.assertNotIn('model.llm.attn_res.attn_queries.4',names)
        self.assertIn('model.llm.attn_res.attn_queries.0',mandatory_sliding_sources(1,c))
        self.assertIn('model.llm.layers.4.attn.wq_du.weight',names)
        self.assertEqual(names['model.llm.layers.4.attn.wq_du.weight'],((32,32),))
        self.assertNotIn('model.llm.attn_res.attn_queries.0',mandatory_sliding_sources(0,c))
    def test_fail_closed_missing_and_malformed_tensor(self):
        c=config();names={**{'model.llm.embed.weight':((1234,32),),
                             'model.llm.embed_norm.weight':((32,),)},**mandatory_sliding_sources(0,c)}
        spans={name:Source(shape=opts[0],dtype='BF16') for name,opts in names.items()}
        # This synthetic config does not meet strict real-model schedule; metadata
        # validator must reject instead of reporting success.
        with self.assertRaises((ValueError,KeyError)):
            readiness_from_spans(spans,c,layers=1)
        del spans['model.llm.embed.weight']
        with self.assertRaises(ValueError):
            readiness_from_spans(spans,c,layers=0)

    def test_official_schedule_and_strict_source_preflight(self):
        fixture=HERE.parents[1]/'tests/models/inkling2/fixtures/config.json'
        if not fixture.exists():
            self.skipTest('requires Stage-1 official HF config fixture')
        official=json.loads(fixture.read_text())
        dims=official['text_config']
        expected={'model.llm.embed.weight':((dims['vocab_size'],dims['hidden_size']),),
                  'model.llm.embed_norm.weight':((dims['hidden_size'],),)}
        for layer in range(5):
            expected.update(mandatory_sliding_sources(layer,official))
        spans={name:Source(shape=options[0],dtype='BF16') for name,options in expected.items()}
        result=readiness_from_spans(spans,official,layers=5)
        self.assertFalse(result['full_forward_supported'])
        self.assertEqual(result['first_visit_cache_slots'],[0,1,2,3,4])
        self.assertEqual(result['declared_sources'],len(expected))
        changed=dict(spans)
        changed['model.llm.layers.4.attn.wq_du.weight']=Source(shape=(4,4),dtype='BF16')
        with self.assertRaisesRegex(ValueError,'unsupported source geometry'):
            readiness_from_spans(changed,official,layers=5)

@unittest.skipIf(torch is None,'requires torch')
class TestRealMath(unittest.TestCase):
    def setUp(self):
        self.g=Geometry(hidden=32,intermediate=16,query_heads=4,kv_heads=2,
                        head_dim=8,rel_dim=4,rel_extent=8,experts=16,top_k=8,conv_kernel=4)
        self.fake=FakeCarrier()
        self.weights=TorchCarrierWeights(self.fake,'cpu',self.g)

    def test_nvfp4_real_source_words(self):
        part=self.fake.bank(0,'w2',[1])[0]
        a=decode_expert(part,rows=32,cols=16,fmt='nvfp4',device=torch.device('cpu'))
        self.assertTrue(bool((a==1).all()))
    def test_nvfp4_rejects_nan_and_bad_stride(self):
        part=self.fake.bank(0,'w2',[1])[0]
        with self.assertRaises(ValueError):
            decode_expert(Source(codes=part.codes,scales=b'\x7f'*(32),amax=1.,denominator=1.),
                          rows=32,cols=16,fmt='nvfp4',device=torch.device('cpu'))
        with self.assertRaises(ValueError):
            decode_expert(part,rows=32,cols=17,fmt='nvfp4',device=torch.device('cpu'))
    def test_mxfp8_e4m3_ue8m0_zero_word(self):
        part=Source(codes=bytes([0x38])*32,scales=b'\x7f')
        a=decode_expert(part,rows=1,cols=32,fmt='mxfp8',device=torch.device('cpu'))
        self.assertTrue(torch.equal(a,torch.ones((1,32))))
        with self.assertRaises(ValueError):
            decode_expert(Source(codes=bytes(32),scales=b'\xff'),rows=1,cols=32,
                          fmt='mxfp8',device=torch.device('cpu'))
    def test_single_real_style_layer_and_top8_count(self):
        hidden=self.weights.embedding(7)
        blocks=torch.empty(1,1,0,32)
        out,blocks,record=sliding_one_token_layer(self.weights,hidden,blocks,0)
        self.assertEqual(tuple(out.shape),(1,1,32))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(tuple(blocks.shape),(1,1,1,32))
        self.assertEqual(len(record['top8_expert_ids']),8)
        self.assertEqual(len({*record['top8_expert_ids']}),8)
        self.assertEqual(self.fake.locs[0][1],'w13')
        self.assertEqual(len(self.fake.locs[0][2]),8)
    def test_no_silent_bad_position_or_full_layer(self):
        hidden=self.weights.embedding(7);blocks=torch.empty(1,1,0,32)
        for bad in (-1,5,23):
            with self.assertRaises(ValueError):
                sliding_one_token_layer(self.weights,hidden,blocks,bad)
        with self.assertRaises(ValueError):
            sliding_one_token_layer(self.weights,hidden,blocks,0,position=1)
    def test_stage5i_prelude_composes_same_single_layer(self):
        h1,b1,t1=run_prelude(self.weights,4,1)
        w2=TorchCarrierWeights(FakeCarrier(),'cpu',self.g)
        h3,b3,t3=run_prelude(w2,4,3)
        self.assertTrue(torch.equal(b1,b3))
        self.assertEqual([x['physical_layer'] for x in t3],[0,1,2])
        self.assertEqual(tuple(h3.shape),(1,1,32))
        self.assertGreater(float((h3-h1).abs().max()),0.)
        self.assertTrue(bool(torch.isfinite(h3).all()))
    def test_stage5i_rejects_unimplemented_full_forward(self):
        for value in (0,6,24,48,-1):
            with self.assertRaises(ValueError):run_prelude(self.weights,4,value)
    @unittest.skipIf(torch is None or not torch.cuda.is_available(),'requires CUDA')
    def test_stage5i_toy_cuda_matches_cpu(self):
        cpu=TorchCarrierWeights(FakeCarrier(),'cpu',self.g)
        gpu=TorchCarrierWeights(FakeCarrier(),'cuda:0',self.g)
        x,_,_=run_prelude(cpu,18,2)
        y,_,_=run_prelude(gpu,18,2)
        self.assertLess(float((x-y.cpu()).abs().max()),1e-4)

    def test_trace_reproducible(self):
        w1=TorchCarrierWeights(FakeCarrier(),'cpu',self.g)
        w2=TorchCarrierWeights(FakeCarrier(),'cpu',self.g)
        a,b,x=run_prelude(w1,18,2)
        c,d,y=run_prelude(w2,18,2)
        self.assertTrue(torch.equal(a,c))
        self.assertEqual(x,y)

if __name__=='__main__':unittest.main()

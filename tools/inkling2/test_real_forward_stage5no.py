"""Stage 5N/O: 48 original-weight-style steps and bounded padded logit head."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from run_real_forward_stage5no import postlude_sources,full_forward_preflight

try:
    import torch
    from real_postlude_stage5no import (
        run_real_full_one_token,postlude_one_token_layer,finish_hidden,
        _bounded_logit_scan,streaming_unembed_topk,
    )
    import test_real_core_stage5lm as _fixtures
    from real_core_stage5lm import run_real_core
except ImportError:
    torch=None


def official():
    path=HERE.parents[1]/'tests/models/inkling2/fixtures/config.json'
    return json.loads(path.read_text()) if path.exists() else None


def all_expected(config):
    from run_real_sliding_stage5hi import mandatory_sliding_sources
    from run_real_block_stage5jk import mandatory_full_sources
    from run_real_core_stage5lm import core_source_shapes
    t=config['text_config'];d=t['hidden_size']
    needed={'model.llm.embed.weight':((t['vocab_size'],d),),
            'model.llm.embed_norm.weight':((d,),)}
    for layer in range(5):needed.update(mandatory_sliding_sources(layer,config))
    needed.update(mandatory_full_sources(config))
    for layer in range(6,18):needed.update(core_source_shapes(layer,config))
    for layer in range(6,18):
        for role in ('attn','k','v','mlp'):
            prefix=f'model.llm.layers.{layer}.controllers.{role}.'
            needed.update({prefix+'down_weight':((64,2*d),),
                           prefix+'forget_weight':((d,64),),
                           prefix+'gain_weight':((d,64),)})
    needed['model.llm.middle_cycle_cross_pass_queries']=((24,d),)
    needed['model.llm.middle_cycle_cross_pass_kv_queries']=((24,d),)
    for layer in range(18,24):needed.update(postlude_sources(layer,config))
    needed['model.llm.attn_res.output_query']=((d,),)
    needed['model.llm.norm.weight']=((d,),)
    needed['model.llm.unembed.weight']=((t['vocab_size'],d),)
    return needed


class TestMetadata(unittest.TestCase):
    def test_postlude_sliding_and_final_full_exact_contracts(self):
        cfg=official()
        if cfg is None:self.skipTest('official fixture unavailable')
        for layer in range(18,23):
            shapes=postlude_sources(layer,cfg)
            self.assertEqual(shapes[f'model.llm.attn_res.attn_queries.{layer-1}'],((2304,),))
            self.assertIn(f'model.llm.layers.{layer}.attn.k_sconv.weight',shapes)
            self.assertNotIn(f'model.llm.layers.{layer}.attn.k_sconv.static_w',shapes)
        full=postlude_sources(23,cfg)
        self.assertEqual(full['model.llm.attn_res.attn_queries.22'],((3,2304),))
        self.assertEqual(full['model.llm.layers.23.attn.k_sconv.static_w'],((4,1024),))
        for bad in (17,24,True,23.0):
            with self.assertRaises(ValueError):postlude_sources(bad,cfg)

    def test_all_48_visits_exact_slots_metadata(self):
        cfg=official()
        if cfg is None:self.skipTest('official fixture unavailable')
        shape=all_expected(cfg)
        spans={name:SimpleNamespace(dtype='BF16' if '.controllers.' in name or name.endswith('unembed.weight') else 'F32',shape=allowed[0])
               for name,allowed in shape.items()}
        plan=full_forward_preflight(spans,cfg,'greedy')
        self.assertEqual(plan['logical_visits'],48)
        self.assertEqual(plan['postlude_physical_layers'],list(range(18,24)))
        self.assertEqual(plan['cache_slots'][-6:],list(range(18,24)))
        self.assertEqual(len(set(plan['cache_slots'])),48)
        self.assertEqual(plan['unpadded_vocab_size'],200058)
        self.assertEqual(plan['quantization_postlude']['mxfp8'],[20,21,22,23])
        missing=dict(spans);del missing['model.llm.attn_res.attn_queries.22']
        with self.assertRaisesRegex(ValueError,'unsupported HF tensor geometry'):
            full_forward_preflight(missing,cfg,'hidden')
        wrong=dict(spans)
        wrong['model.llm.unembed.weight']=SimpleNamespace(dtype='BF16',shape=(200058,2304))
        with self.assertRaisesRegex(ValueError,'unsupported HF tensor geometry'):
            full_forward_preflight(wrong,cfg,'greedy')
        plan_hidden=full_forward_preflight({k:v for k,v in spans.items() if k!='model.llm.unembed.weight'},cfg,'hidden')
        self.assertEqual(plan_hidden['mode'],'hidden')
        self.assertEqual(plan_hidden['logical_visits'],48)

    def test_unsupported_revision_and_mode_rejected(self):
        cfg=official()
        if cfg is None:self.skipTest('official fixture unavailable')
        bad=json.loads(json.dumps(cfg));bad['text_config']['vocab_size']=200058
        with patch('run_real_forward_stage5no.core_preflight',return_value={'cache_slots': []}):
            with self.assertRaisesRegex(ValueError,'unexpected padded/unpadded vocabulary'):
                full_forward_preflight({},bad,'hidden')
        with self.assertRaises(ValueError):full_forward_preflight({},cfg,'not_a_mode')


@unittest.skipIf(torch is None,'requires PyTorch')
class TestTensorIntegration(unittest.TestCase):
    @staticmethod
    def small_weights():
        weights=_fixtures.TestCoreNumerics.make_weights()
        carrier=weights.reader
        originals=dict(carrier.params)
        for layer in range(18,24):
            template=11 if layer==23 else 16
            old=f'model.llm.layers.{template}.'
            new=f'model.llm.layers.{layer}.'
            for name,value in originals.items():
                if name.startswith(old):
                    carrier.params[new+name[len(old):]]=value.clone()
                elif name==f'model.llm.attn_res.attn_queries.{template-1}':
                    carrier.params[f'model.llm.attn_res.attn_queries.{layer-1}']=value.clone()
                elif name==f'model.llm.attn_res.mlp_queries.{template}':
                    carrier.params[f'model.llm.attn_res.mlp_queries.{layer}']=value.clone()
        carrier.params['model.llm.attn_res.output_query']=torch.arange(32,dtype=torch.float32)/100
        carrier.params['model.llm.norm.weight']=torch.ones(32)
        # The inherited compact synthetic fixture has intermediate=16, which
        # cannot be encoded as MXFP8 G32 for W2. Its actual MXFP8 path was
        # already validated with real Inkling dimensions in Stage 5G.
        # Here we test postlude/trace/control-flow with deterministic MoE stubs
        # only for physical layers 20..23.
        original_moe=weights.moe
        def compact_postlude_moe(hidden,layer):
            if layer>=20:return torch.zeros_like(hidden),list(range(8))
            return original_moe(hidden,layer)
        weights.moe=compact_postlude_moe
        return weights

    def test_postlude_block_boundary_append_and_full_attention(self):
        w=self.small_weights()
        hidden,blocks,_=run_real_core(w,9,core_layers=12,passes=3,gate_rank=4)
        before=hidden.clone()
        x,new_blocks,record=postlude_one_token_layer(w,hidden,blocks,18)
        self.assertEqual(tuple(new_blocks.shape),(1,1,4,32))
        self.assertTrue(torch.equal(new_blocks[:,:,3,:],before))
        self.assertEqual(record['cache_slot'],18)
        for layer in range(19,24):
            x,new_blocks,record=postlude_one_token_layer(w,x,new_blocks,layer)
        self.assertEqual(record['attention_type'],'full')
        self.assertTrue(record['dynamic_kv_convolution'])
        self.assertTrue(torch.isfinite(finish_hidden(w,x,new_blocks)).all())

    def test_full_48_trace_and_determinism(self):
        w=self.small_weights()
        hidden,blocks,trace=run_real_full_one_token(w,9,gate_rank=4)
        self.assertEqual(len(trace),48)
        self.assertEqual(len(set(r['cache_slot'] for r in trace)),48)
        self.assertEqual([r['physical_layer'] for r in trace[-6:]],list(range(18,24)))
        self.assertEqual([r['attention_type'] for r in trace[-6:]],['sliding']*5+['full'])
        self.assertEqual(tuple(blocks.shape),(1,1,4,32))
        self.assertTrue(torch.isfinite(hidden).all())
        other,_,_=run_real_full_one_token(self.small_weights(),9,gate_rank=4)
        self.assertTrue(torch.equal(hidden,other))

    def test_postlude_input_contracts(self):
        w=self.small_weights()
        zero=torch.zeros(1,1,32)
        three=torch.zeros(1,1,3,32)
        with self.assertRaises(ValueError):postlude_one_token_layer(w,zero,three,17)
        with self.assertRaises(ValueError):postlude_one_token_layer(w,zero,three,19)
        with self.assertRaises(ValueError):finish_hidden(w,zero,three)


@unittest.skipIf(torch is None,'requires PyTorch')
class TestLogits(unittest.TestCase):
    class FakeArt:
        def __init__(self,weight,fmt):
            self.matrix=weight.contiguous()
            self.obj=SimpleNamespace(kind='tensor',format=fmt,shape=tuple(weight.shape),offset=0)
            self.raw=weight.view(torch.uint8).numpy().tobytes()
            self.chunks=[]
        def object(self,name):
            if name!='hf/model.llm.unembed.weight':raise KeyError(name)
            return self.obj
        def read_range(self,offset,length):
            self.chunks.append((offset,length))
            return self.raw[offset:offset+length]

    def test_output_scan_excludes_padding_and_matches_dense_softcap(self):
        torch.manual_seed(1234)
        for dtype,fmt in ((torch.bfloat16,'bf16'),(torch.float32,'fp32')):
            weights=torch.randn(11,4,dtype=torch.float32).to(dtype)
            weights[-3:,:]=torch.tensor([200.,100.,100.,100.]).to(dtype)
            art=self.FakeArt(weights,fmt)
            x=torch.tensor([[[1.,2.,3.,4.]]])
            top=_bounded_logit_scan(art,x,shape=(11,4),unpadded_vocab=8,
                softcap=30.,topk=3,chunk_rows=3)
            dense=(weights[:8].float()@x.reshape(4)).float()
            dense=30*torch.tanh(dense/30)
            expected=sorted([(float(v),i) for i,v in enumerate(dense)],key=lambda p:(-p[0],p[1]))[:3]
            self.assertEqual([a['token_id'] for a in top],[i for _,i in expected])
            self.assertTrue(all(abs(a['logit']-v)<1e-5 for a,(v,_) in zip(top,expected)))
            self.assertTrue(all(i+n<=8*4*(2 if fmt=='bf16' else 4) for i,n in art.chunks))
            self.assertEqual(len(art.chunks),3)


    def test_real_ninfer_v3_reader_streams_small_unembedding(self):
        try:
            from tools.artifact.schema import TensorSpec
            from tools.artifact.writer import ArtifactWriter
            from tools.artifact.reader import Artifact
        except ImportError:
            self.skipTest('requires complete NInfer-v3 artifact writer/reader in worktree')
        torch.manual_seed(44)
        rows=torch.randn(11,4,dtype=torch.float32).to(torch.bfloat16)
        rows[-3:]=torch.tensor([300.,300.,300.,300.]).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'short-v3.ninfer'
            spec=TensorSpec('hf/model.llm.unembed.weight',(11,4),'bf16','contiguous_le_v1')
            with ArtifactWriter(path,[spec],components={'text':{'config':{}}},bindings={}) as writer:
                writer.write_object(spec.id,rows.contiguous().view(torch.uint8).numpy().tobytes())
            with Artifact(path) as artifact:
                result=_bounded_logit_scan(artifact,torch.tensor([[[1.,2.,3.,4.]]]),
                    shape=(11,4),unpadded_vocab=8,softcap=30.,topk=3,chunk_rows=2)
                expected=torch.mv(rows[:8].float(),torch.tensor([1.,2.,3.,4.]))
                expected=30*torch.tanh(expected/30)
                wanted=torch.topk(expected,3).indices.tolist()
                self.assertEqual([i['token_id'] for i in result],wanted)
                self.assertTrue(all(i['token_id']<8 for i in result))

    def test_topk_is_chunk_size_invariant(self):
        torch.manual_seed(77)
        rows=torch.randn(11,4,dtype=torch.float32)
        x=torch.tensor([[[1.,-.5,2.,.25]]])
        a=_bounded_logit_scan(self.FakeArt(rows,'fp32'),x,shape=(11,4),
                              unpadded_vocab=8,softcap=None,topk=5,chunk_rows=1)
        b=_bounded_logit_scan(self.FakeArt(rows,'fp32'),x,shape=(11,4),
                              unpadded_vocab=8,softcap=None,topk=5,chunk_rows=7)
        self.assertEqual(a,b)

    def test_saturated_ties_choose_first_token(self):
        rows=torch.ones(10,4,dtype=torch.float32)*1000
        x=torch.ones(1,1,4)
        winners=_bounded_logit_scan(self.FakeArt(rows,'fp32'),x,
            shape=(10,4),unpadded_vocab=7,softcap=30.,topk=3,chunk_rows=2)
        self.assertEqual([w['token_id'] for w in winners],[0,1,2])

    def test_logit_rejection(self):
        m=torch.ones(5,4,dtype=torch.bfloat16)
        art=self.FakeArt(m,'bf16')
        with self.assertRaises(ValueError):
            _bounded_logit_scan(art,torch.ones(1,1,3),shape=(5,4),unpadded_vocab=4,softcap=30,topk=1,chunk_rows=1)
        with self.assertRaises(ValueError):
            _bounded_logit_scan(art,torch.ones(1,1,4),shape=(5,4),unpadded_vocab=6,softcap=30,topk=1,chunk_rows=1)
        with self.assertRaises(ValueError):
            streaming_unembed_topk(SimpleNamespace(nonexpert=art),torch.ones(1,1,4))


if __name__=='__main__': unittest.main()

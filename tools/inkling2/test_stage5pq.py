"""Small independent synthetic tests for Stage5P (comparison) and Stage5Q (stateful)."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


@unittest.skipUnless(np is not None, 'NumPy required for Stage5P report tests')
class TestStage5P(unittest.TestCase):
    def setUp(self):
        from parity_stage5p import execution_layers, execution_slots
        self.layers=execution_layers();self.slots=execution_slots()

    def test_official_visit_order_and_slot_mapping(self):
        self.assertEqual(len(self.layers),48)
        self.assertEqual(self.layers[:18],list(range(18)))
        self.assertEqual(self.layers[18:30],list(range(6,18)))
        self.assertEqual(self.layers[30:42],list(range(6,18)))
        self.assertEqual(self.layers[42:],list(range(18,24)))
        self.assertEqual(self.slots[:18],list(range(18)))
        self.assertEqual(self.slots[18:42],list(range(24,48)))
        self.assertEqual(self.slots[42:],list(range(18,24)))
        self.assertEqual(set(self.slots),set(range(48)))

    def _samples(self,p,alter=None):
        from parity_stage5p import _save
        v=np.ones((48,2304),dtype=np.float32)*0.5
        if alter:alter(v)
        _save(p/'s.npz',role='student',revision='abc',token=1000,
              visits=v,final_hidden=v[-1],ids=np.array([5,1,2],dtype=np.int32),
              logits=np.array([4,3,2],dtype=np.float32),framework='fake student')
        _save(p/'h.npz',role='hf',revision='abc',token=1000,
              visits=np.ones((48,2304),dtype=np.float32)*0.5,
              final_hidden=np.ones(2304,dtype=np.float32)*0.5,
              ids=np.array([5,1,2],dtype=np.int32),
              logits=np.array([4,3,2],dtype=np.float32),framework='fake official')

    def test_comparison_success(self):
        from parity_stage5p import compare
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._samples(p)
            result=compare(p/'s.npz',p/'h.npz',atol=0.01,rtol=0.01)
            self.assertEqual(result['status'],'PASS')
            self.assertIsNone(result['first_failing_visit'])
            self.assertEqual(len(result['visits']),48)

    def test_first_failing_visit_identified(self):
        from parity_stage5p import compare
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._samples(p,lambda a:a.__setitem__((27,111),0.9))
            result=compare(p/'s.npz',p/'h.npz',atol=0.01,rtol=0.01)
            self.assertEqual(result['first_failing_visit'],27)
            self.assertEqual(result['visits'][27]['cache_slot'],33)
            self.assertEqual(result['status'],'MISMATCH')
            self.assertTrue(result['same_top1'])

    def test_revision_mismatch_fails_closed(self):
        from parity_stage5p import compare, _save
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._samples(p)
            with np.load(p/'h.npz',allow_pickle=False) as z:
                fields={k:z[k].copy() for k in z.files}
            fields['revision']=np.array('not-abc')
            np.savez_compressed(p/'h2.npz',**fields)
            with self.assertRaises(ValueError):
                compare(p/'s.npz',p/'h2.npz',atol=1,rtol=1)

    def test_corrupt_or_missing_role_rejected(self):
        from parity_stage5p import _save,compare
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._samples(p)
            with self.assertRaises(FileExistsError):
                _save(p/'s.npz',role='student',revision='abc',token=1000,
                      visits=np.zeros((48,2304),np.float32),
                      final_hidden=np.zeros(2304,np.float32),ids=np.array([1]),
                      logits=np.array([1.]),framework='fake')
            with self.assertRaises(ValueError):
                compare(p/'s.npz',p/'s.npz',atol=.01,rtol=.01)

    def test_different_top1_never_claims_pass(self):
        from parity_stage5p import compare
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self._samples(p)
            with np.load(p/'s.npz',allow_pickle=False) as z:
                fields={k:z[k].copy() for k in z.files}
            fields['top_ids'][0]=3
            np.savez_compressed(p/'s2.npz',**fields)
            result=compare(p/'s2.npz',p/'h.npz',atol=1,rtol=1)
            self.assertEqual(result['status'],'MISMATCH')
            self.assertFalse(result['same_top1'])


class TinyWeights:
    """Deterministic fake weights, independent of the production checkpoint reader."""
    def __init__(self):
        from real_sliding_stage5h import Geometry
        self.geom=Geometry(hidden=32,intermediate=16,query_heads=2,kv_heads=1,
                           head_dim=8,rel_dim=4,rel_extent=8,experts=4,top_k=2,conv_kernel=4)
        self.device=torch.device('cpu')
        self._values={}

    def tensor(self,name,shape):
        old=self._values.get(name)
        if old is not None:
            if old.shape != shape:raise ValueError('inconsistent requested shape')
            return old
        seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
        gen=torch.Generator().manual_seed(seed)
        if (name.endswith(('norm.weight','norm.scale')) or
                name.endswith(('q_l2_clip.scale','rel_l2_clip.scale'))):
            v=torch.ones(shape)
        elif 'controllers.' in name:
            v=torch.zeros(shape)
        else:
            v=torch.randn(shape,generator=gen)*0.02
        self._values[name]=v
        return v

    def conv(self,name,channels):
        return self.tensor(name,(channels,self.geom.conv_kernel))

    def embedding(self,token):
        return torch.arange(self.geom.hidden,dtype=torch.float32).reshape(1,1,-1)*.003+token*.0001

    def moe(self,hidden,layer):
        return torch.tanh(hidden)*.01, [0,1]

    def clear(self):
        pass


@unittest.skipUnless(torch is not None, 'PyTorch required for Stage5Q tensor tests')
class TestStage5Q(unittest.TestCase):
    def test_slots_and_illegal_requests(self):
        from streaming_stage5q import slot_for,IncrementalCache48
        self.assertEqual([slot_for(i,0) for i in range(24)],list(range(24)))
        self.assertEqual([slot_for(i,1) for i in range(6,18)],list(range(24,36)))
        self.assertEqual([slot_for(i,2) for i in range(6,18)],list(range(36,48)))
        for physical,passnum in ((-1,0),(24,0),(5,1),(18,1),(17,3)):
            with self.assertRaises(ValueError):slot_for(physical,passnum)
        c=IncrementalCache48()
        with self.assertRaises(ValueError):c.end()
        with self.assertRaises(ValueError):c.begin(1)
        c.begin(0)
        with self.assertRaises(ValueError):c.begin(0)
        with self.assertRaises(ValueError):c.end()
        c.abort()
        with self.assertRaises(ValueError):c.begin(0)

    def test_static_and_dynamic_one_token_oracles(self):
        from streaming_stage5q import IncrementalCache48
        from forward_oracle_stage5a import static_short_convolution,dynamic_short_convolution
        c=IncrementalCache48()
        c.begin(0)
        g=torch.Generator().manual_seed(123)
        x=torch.randn((1,1,8),generator=g)
        taps=torch.randn((8,4),generator=g)
        got=c.static_conv(0,0,x,taps)
        expected=static_short_convolution(x,taps)
        self.assertTrue(torch.allclose(got,expected,atol=1e-6))
        q=torch.randn((1,1,6),generator=g)
        dtaps=torch.randn((4,8),generator=g)
        proj=torch.randn((2*4,6),generator=g)
        y=c.dynamic_conv(0,1,x,q,dtaps,proj,heads=2)
        ref=dynamic_short_convolution(x,q,dtaps,proj,num_heads=2)
        self.assertTrue(torch.allclose(y,ref,atol=1e-6))
        with self.assertRaises(ValueError):c.static_conv(0,0,x,taps)

    def test_three_token_conv_prefill_vs_decode_independent(self):
        from streaming_stage5q import IncrementalCache48
        from forward_oracle_stage5a import static_short_convolution,dynamic_short_convolution
        generator=torch.Generator().manual_seed(912)
        x=torch.randn((1,3,8),generator=generator)
        dynamic_input=torch.randn((1,3,6),generator=generator)
        static_taps=torch.randn((8,4),generator=generator)
        dynamic_taps=torch.randn((4,8),generator=generator)
        projection=torch.randn((8,6),generator=generator)
        c=IncrementalCache48(kernel=4)
        all_static=[];all_dynamic=[]
        with torch.inference_mode():
            for position in range(3):
                c.begin(position)
                for slot in range(48):
                    for path in range(4):
                        if slot==1 and path==0:
                            output=c.dynamic_conv(slot,path,x[:,position:position+1],
                                dynamic_input[:,position:position+1],dynamic_taps,projection,heads=2)
                            all_dynamic.append(output)
                        else:
                            output=c.static_conv(slot,path,x[:,position:position+1],static_taps)
                            if slot==0 and path==0:all_static.append(output)
                    q=torch.ones((1,2,1,4))
                    key=value=torch.ones((1,1,1,4))
                    c.attention(slot,q,key,value,position,sliding_window=2,
                        bias_from_distances=lambda d:torch.zeros((1,2,1,d.shape[-1])),
                        sinks=torch.zeros((1,2,1,1)))
                c.end()
        self.assertTrue(torch.allclose(torch.cat(all_static,dim=1),
                        static_short_convolution(x,static_taps),atol=1e-6))
        self.assertTrue(torch.allclose(torch.cat(all_dynamic,dim=1),
                        dynamic_short_convolution(x,dynamic_input,dynamic_taps,projection,
                                                  num_heads=2),atol=1e-6))
        self.assertEqual(c.key_positions[0],[1,2])
        self.assertEqual(c.key_positions[1],[1,2])

    def test_48_visit_position0_parity_and_position1(self):
        from real_postlude_stage5no import run_real_full_one_token
        from streaming_stage5q import IncrementalCache48,forward_one_token
        w=TinyWeights()
        with torch.inference_mode():
            old,blocks,old_trace=run_real_full_one_token(w,1000,gate_rank=4)
            cache=IncrementalCache48()
            new,trace=forward_one_token(w,cache,1000,gate_rank=4)
            self.assertTrue(torch.equal(new,old),float(torch.max(torch.abs(new-old))))
            self.assertEqual(cache.next_position,1)
            self.assertEqual(len(cache.kv),48)
            self.assertEqual(len(cache.conv),192)
            self.assertEqual(len(trace),48)
            self.assertEqual([i['cache_slot'] for i in trace],
                             list(range(18))+list(range(24,48))+list(range(18,24)))
            follow,record=forward_one_token(w,cache,1001,gate_rank=4)
            self.assertTrue(torch.isfinite(follow).all())
            self.assertEqual(cache.next_position,2)
            self.assertEqual(len(cache.kv),48)
            self.assertEqual(len(cache.conv),192)
            self.assertEqual(cache.key_positions[0],[0,1])
            self.assertEqual(cache.key_positions[24],[0,1])
            self.assertEqual(cache.key_positions[36],[0,1])
            self.assertEqual(len(record),48)

    def test_attention_sink_and_slot_isolation(self):
        from streaming_stage5q import IncrementalCache48
        c=IncrementalCache48();c.begin(0)
        q=torch.ones((1,2,1,4))
        k=torch.ones((1,1,1,4))
        v=torch.tensor([[[[2.,-1.,3.,4.]]]])
        sink=torch.zeros((1,2,1,1))
        out=c.attention(0,q,k,v,0,sliding_window=2,
            bias_from_distances=lambda d:torch.zeros((1,2,1,d.shape[-1])),sinks=sink)
        self.assertEqual(out.shape,(1,1,2,4))
        self.assertEqual(c.key_positions[0],[0])
        self.assertNotIn(1,c.kv)
        with self.assertRaises(ValueError):c.attention(0,q,k,v,0,sliding_window=2,
            bias_from_distances=lambda d:torch.zeros((1,2,1,d.shape[-1])),sinks=sink)

    def test_fail_closed_during_step(self):
        from streaming_stage5q import IncrementalCache48,forward_one_token
        w=TinyWeights();cache=IncrementalCache48()
        def invalid(*a,**kw):raise ValueError('failing expert intentionally')
        w.moe=invalid
        with torch.inference_mode(),self.assertRaisesRegex(ValueError,'failing expert'):
            forward_one_token(w,cache,1000,gate_rank=4)
        self.assertTrue(cache.invalid)
        with self.assertRaises(ValueError):cache.begin(0)

    def test_student_hook_rejects_incompatible_model_shape_and_restores(self):
        from parity_stage5p import TraceRecorder,_capture_student
        from real_postlude_stage5no import run_real_full_one_token
        import real_full_stage5j
        original=real_full_stage5j.sliding_one_token_layer
        w=TinyWeights();record=TraceRecorder([],[])
        with torch.inference_mode(),self.assertRaisesRegex(ValueError,'2304'),_capture_student(record):
            run_real_full_one_token(w,1000,gate_rank=4)
        self.assertIs(real_full_stage5j.sliding_one_token_layer,original)



@unittest.skipUnless(torch is not None, 'PyTorch required for Stage5P real-shape hook')
class TestStage5PRealShapes(unittest.TestCase):
    def test_recorder_rejects_wrong_hidden_size(self):
        from parity_stage5p import TraceRecorder
        rec=TraceRecorder([],[])
        with self.assertRaises(ValueError):rec.add(0,torch.ones((1,1,32)))
        self.assertEqual(len(rec.layers),0)

    def test_recorder_48_2304_values(self):
        from parity_stage5p import TraceRecorder
        rec=TraceRecorder([],[])
        sample=torch.ones((1,1,2304))
        for physical in [*range(18),*range(6,18),*range(6,18),*range(18,24)]:
            rec.add(physical,sample)
        self.assertEqual(rec.matrix().shape,(48,2304))
        with self.assertRaises(ValueError):rec.add(23,sample)


if __name__ == '__main__':unittest.main()

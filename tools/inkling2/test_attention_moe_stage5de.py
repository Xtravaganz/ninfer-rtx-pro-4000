"""Independent stdlib scalar contracts for isolated Stage 5D/5E CUDA backends.

They do not pretend that the CUDA test passed; run the Docker native tests too.
"""
from __future__ import annotations

import math
import random
import unittest


def slot(layer: int, pass_id: int) -> int:
    if not 0 <= layer < 24 or not 0 <= pass_id < 3:
        raise ValueError('invalid physical layer/pass')
    if not 6 <= layer < 18:
        if pass_id:
            raise ValueError('non-core layers execute once')
        return layer
    return layer if not pass_id else 24 + (pass_id - 1) * 12 + layer - 6


def softmax_with_sink(query, key_history, value_history, sink, relative, valid,
                      sequence_ids, position, window=0):
    dim = len(query)
    logits = []
    for token, k in enumerate(key_history):
        if (token > position or not valid[token] or
                sequence_ids[token] != sequence_ids[position] or
                (window and position-token >= window)):
            continue
        dot = math.fsum(a*b for a, b in zip(query, k)) / math.sqrt(dim)
        distance = position-token
        bias = relative[distance] if distance < len(relative) else 0.0
        logits.append((token, dot+bias))
    maximum = max([sink, *[logit for _,logit in logits]])
    denominator = math.exp(sink-maximum) + math.fsum(math.exp(x-maximum) for _, x in logits)
    return [math.fsum(math.exp(logit-maximum) * value_history[t][i] for t,logit in logits)/denominator
            for i in range(dim)]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def select_top8(scores, corrections, route_scale):
    if len(scores)!=128 or len(corrections)!=128:
        raise ValueError('routing requires 128 experts')
    indices=sorted(range(128),key=lambda i:(-(scores[i]+corrections[i]),i))[:8]
    total=math.fsum(scores[i] for i in indices)
    weights=[route_scale*scores[i]/(total+1e-20) for i in indices]
    return indices,weights


def softcap_swiglu(raw_gate, raw_up):
    return 4 * math.tanh(raw_gate / 4) * sigmoid(raw_gate) * 25 * math.tanh(raw_up / 25)


def decode_nvfp4_pair(code: int, e4m3scale: int, amax: float, denominator: float):
    if not 0 <= code < 256 or not 0 <= e4m3scale < 127 or amax <= 0 or denominator <= 0:
        raise ValueError('invalid NVFP4 words')
    base=(0, .5, 1, 1.5, 2, 3, 4, 6)
    exp=(e4m3scale >> 3)&15
    mant=e4m3scale&7
    s=math.ldexp(mant, -9) if exp==0 else math.ldexp(8+mant, exp-10)
    return tuple((-base[n&7] if n&8 else base[n&7])*s*amax/denominator for n in (code&15,code>>4))


def decode_mxfp8(word, ue8):
    if not 0 <= word <= 255 or word&127==127 or not 0 <= ue8 < 255:
        raise ValueError('invalid MXFP8 words')
    exp=(word>>3)&15;man=word&7
    v=math.ldexp(man,-9) if exp==0 else math.ldexp(8+man,exp-10)
    return (-v if word&128 else v)*math.ldexp(1.,ue8-127)


class TestStage5D(unittest.TestCase):
    def test_48_unique_logical_slots(self):
        physical = [(p,0) for p in range(24)] + [(p,r) for r in (1,2) for p in range(6,18)]
        self.assertEqual(sorted(slot(*pair) for pair in physical),list(range(48)))
        with self.assertRaises(ValueError):slot(18,1)

    def test_sink_can_absorb_every_logit(self):
        self.assertEqual(softmax_with_sink([1.],[ [2.] ],[[3.]], 0.,[0.], [False],[1],0),[0.])
        out=softmax_with_sink([1.],[[0.]],[[2.]],0.,[0.],[True],[1],0)
        self.assertAlmostEqual(out[0],1.0)

    def test_zero_bias_outside_extent(self):
        k=[[.1], [.2], [.3], [.4]];v=[[1.],[2.],[3.],[4.]]
        got=softmax_with_sink([.5],k,v,0.,[.1],[True]*4,[0]*4,3)
        no=softmax_with_sink([.5],k,v,0.,[.0],[True]*4,[0]*4,3)
        self.assertNotEqual(got,no)
        expected=sum(v*math.exp((i+1)*0.05+(0.1 if i==3 else 0)) for i,v in enumerate((1,2,3,4))) / (1+sum(math.exp((i+1)*0.05+(0.1 if i==3 else 0)) for i in range(4)))
        self.assertAlmostEqual(got[0],expected,places=12)

    def test_full_vs_chunk_equivalence(self):
        rnd=random.Random(59)
        k=[[rnd.uniform(-1,1) for _ in range(6)] for _ in range(13)]
        v=[[rnd.uniform(-1,1) for _ in range(6)] for _ in range(13)]
        q=[[rnd.uniform(-1,1) for _ in range(6)] for _ in range(13)]
        valid=[t != 3 for t in range(13)]
        ids=[t//5 for t in range(13)]
        for window in (0,3,9):
            direct=[softmax_with_sink(q[t],k,v,.1,[.03,-.02],valid,ids,t,window) for t in range(13)]
            incremental=[]
            for a,b in ((0,2),(2,7),(7,9),(9,13)):
                incremental += [softmax_with_sink(q[t],k[:b],v[:b],.1,[.03,-.02],valid[:b],ids[:b],t,window) for t in range(a,b)]
            for left,right in zip(direct,incremental):
                for aa,bb in zip(left,right):self.assertAlmostEqual(aa,bb,places=13)

    def test_sequence_boundaries_and_sliding_window(self):
        q=[1.];k=[[0.]]*4;v=[[1.],[3.],[5.],[7.]]
        allout=softmax_with_sink(q,k,v,0.,[],[True]*4,[0]*4,3)
        short=softmax_with_sink(q,k,v,0.,[],[True]*4,[0]*4,3,2)
        reset=softmax_with_sink(q,k,v,0.,[],[True]*4,[0,0,1,1],3)
        self.assertTrue(allout[0] < short[0]);self.assertAlmostEqual(short[0],reset[0])


class TestStage5E(unittest.TestCase):
    def test_correction_bias_changes_indices_not_weight_math(self):
        scores=[.25+.001*i for i in range(128)]
        bias=[3. if i < 8 else -3. for i in range(128)]
        ids,weights=select_top8(scores,bias,8.0)
        self.assertEqual(set(ids),set(range(8)))
        self.assertAlmostEqual(sum(weights),8.)
        for i,w in zip(ids,weights):self.assertAlmostEqual(w,8*scores[i]/sum(scores[j] for j in ids))

    def test_nvfp4_code_and_ratio(self):
        self.assertEqual(decode_nvfp4_pair(0xB2,0x38,1.5,1.0),(1.5,-2.25))
        with self.assertRaises(ValueError):decode_nvfp4_pair(0xB2,0x7F,1.5,1.0)
        with self.assertRaises(ValueError):decode_nvfp4_pair(0,0x38,1.0,0.0)

    def test_ue8m0_zero_is_smallest_not_zero(self):
        self.assertEqual(decode_mxfp8(0x38,0),2**-127)
        self.assertEqual(decode_mxfp8(0xb8,127),-1.0)
        with self.assertRaises(ValueError):decode_mxfp8(0x38,255)
        with self.assertRaises(ValueError):decode_mxfp8(0x7f,127)

    def test_swiglu_uses_unmodified_gate_sigmoid(self):
        gate,up=-1.7,2.3
        correct=softcap_swiglu(gate,up)
        wrong=4*math.tanh(gate/4)*sigmoid(4*math.tanh(gate/4))*25*math.tanh(up/25)
        self.assertGreater(abs(correct-wrong),1e-3)

    def test_two_format_selected_moe_toy(self):
        nv=decode_nvfp4_pair(0xB2,0x38,1.5,1.0)
        fp8=(decode_mxfp8(0x38,127),decode_mxfp8(0xb8,127))
        scores=[.5]*128;bias=[2. if i<8 else -2. for i in range(128)]
        ids,weights=select_top8(scores,bias,8.0)
        self.assertEqual(len(ids),8)
        for vals in (nv,fp8):
            outputs=[softcap_swiglu(vals[0]*.7,vals[1]*.7)*vals[0] for _ in ids]
            result=math.fsum(w*v for w,v in zip(weights,outputs))
            self.assertTrue(math.isfinite(result))
            self.assertAlmostEqual(result,8*outputs[0])


if __name__ == '__main__': unittest.main()

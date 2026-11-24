"""Independent numerical/golden tests for Stage 5A (CPU, optional CUDA)."""
from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from forward_oracle_stage5a import (
        unweighted_rmsnorm, weighted_rmsnorm, l2_clip, scaled_l2_clip,
        relative_position_bias, causal_additive_mask, sink_attention,
        gated_output, static_short_convolution, static_short_convolution_step,
        dynamic_short_convolution, block_attention_residual,
        cross_pass_messages, controller_message_gate, schedule_trace, execute_loop_callbacks,
    )


def manual_static(x, w, mask=None):
    """Scalar independent reference; avoids conv1d, unfold and matrix multiplies."""
    b, t, c = x.shape
    k = w.shape[1]
    z = torch.zeros_like(x, dtype=torch.float64)
    for bi in range(b):
        for ti in range(t):
            for ci in range(c):
                acc = x[bi, ti, ci].item()
                for ki in range(k):
                    src = ti - (k - 1 - ki)
                    if src >= 0:
                        m = 1 if mask is None else mask[bi, src].item()
                        acc += x[bi, src, ci].item() * m * w[ci, ki].item()
                z[bi, ti, ci] = acc / math.sqrt(k + 1)
    return z


def manual_dynamic(x, d, s, p, heads, mask=None, ids=None):
    b, t, c = x.shape
    k = s.shape[0]
    out = torch.zeros_like(x, dtype=torch.float64)
    per_head = c // heads
    for bi in range(b):
        for ti in range(t):
            for ci in range(c):
                h = ci // per_head
                acc = x[bi, ti, ci].item()
                for lag in range(k):
                    prev = ti - lag
                    if prev < 0:
                        continue
                    if ids is not None and ids[bi, prev].item() != ids[bi, ti].item():
                        continue
                    m = 1 if mask is None else mask[bi, prev].item()
                    dynamic_tap = sum(
                        p[h*k+lag, n].item() * d[bi, ti, n].item()
                        for n in range(d.shape[-1])
                    )
                    acc += m * x[bi, prev, ci].item() * (s[lag, ci].item() + dynamic_tap)
                out[bi, ti, ci] = acc / math.sqrt(2*k+1)
    return out


def manual_attention(q, k, v, *, bias=None, mask=None, sinks=None):
    """No batched MatMul, softmax or repeat_interleave used."""
    b, hq, nq, d = q.shape
    hkv, nk = k.shape[1:3]
    out = torch.zeros((b, nq, hq, d), dtype=torch.float64)
    for bi in range(b):
        for hi in range(hq):
            kv_h = hi // (hq // hkv)
            for qi in range(nq):
                lg = []
                for ki in range(nk):
                    score = sum(q[bi, hi, qi, j].item() * k[bi, kv_h, ki, j].item() for j in range(d)) / math.sqrt(d)
                    if bias is not None:
                        score += bias[bi, hi, qi, ki].item()
                    if mask is not None:
                        score += mask[bi, 0, qi, ki].item()
                    lg.append(score)
                if sinks is not None:
                    lg.append(sinks[bi, hi, qi, 0].item())
                maximum = max(lg)
                probs = [math.exp(x-maximum) for x in lg]
                probs = [x/sum(probs) for x in probs]
                for j in range(d):
                    out[bi, qi, hi, j] = sum(probs[ki]*v[bi, kv_h, ki, j].item() for ki in range(nk))
    return out


@unittest.skipUnless(torch is not None, 'requires torch; use existing pytorch/pytorch CUDA image')
class TestStage5A(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5101)

    def test_unweighted_and_weighted_rms(self):
        x = torch.tensor([[3., 4., 0., 0.]])
        expected = x / math.sqrt((9+16)/4+1e-6)
        torch.testing.assert_close(unweighted_rmsnorm(x), expected)
        torch.testing.assert_close(weighted_rmsnorm(x, torch.tensor([1., 2., 3., 4.])),
                                   expected*torch.tensor([1., 2., 3., 4.]))

    def test_l2_clip_zero_and_threshold(self):
        x = torch.tensor([[3.,4.],[0.,0.]])
        got = l2_clip(x,2.)
        torch.testing.assert_close(got[0], torch.tensor([1.2,1.6]),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(got[1], torch.zeros(2))

    def test_scaled_log_clip_position_threshold(self):
        x=torch.tensor([[[[3.,4.]],[[3.,4.]]]])
        scales=torch.ones((1,2))
        pos=torch.tensor([[8191, 16383]])
        got=scaled_l2_clip(x,scales,threshold=5.0,positions=pos)
        self.assertAlmostEqual(float(got[0,0,0,0]), 3.0,places=4)
        self.assertAlmostEqual(float(got[0,1,0,0]), 3.0*(1+.1*math.log(2)),places=4)

    def test_relative_bias_distance_and_zero_outside_range(self):
        h=torch.tensor([[[1.,2.],[2.,3.]]])
        linear=torch.eye(2)
        scales=torch.ones((1,2))
        profiles=torch.tensor([[10.,20.,30.],[1.,2.,3.]])
        distances=torch.tensor([[[0,1,3,-1],[2,0,1,4]]])
        got=relative_position_bias(h,linear,scales,profiles,distances,threshold=100.)
        want=torch.tensor([[[[12.,24.,0.,0.],[69.,23.,46.,0.]]]])
        torch.testing.assert_close(got,want)

    def test_relative_bias_full_log_scale(self):
        h=torch.tensor([[[2.,3.]]]); w=torch.tensor([[1.,0.]]); sc=torch.ones(1,1); p=torch.ones(1,4)
        d=torch.tensor([[0,3,4,-1]])
        got=relative_position_bias(h,w,sc,p,d,threshold=100.,positions=torch.tensor([[16383]]))
        factor=1+.1*math.log(2)
        torch.testing.assert_close(got[0,0,0],torch.tensor([2*factor,2*factor,0,0]))

    def test_causal_and_sliding_mask_boundaries(self):
        p=torch.tensor([[2,3]])
        k=torch.arange(5).unsqueeze(0)
        full=causal_additive_mask(p,k)
        window=causal_additive_mask(p,k,sliding_window=2)
        torch.testing.assert_close(full[0,0,0,:3],torch.zeros(3))
        self.assertTrue(torch.isneginf(full[0,0,0,3:]).all())
        self.assertTrue(torch.isneginf(window[0,0,0,0]))
        torch.testing.assert_close(window[0,0,0,1:3],torch.zeros(2))
        self.assertTrue(torch.isneginf(window[0,0,1,1]))

    def test_sink_dilutes_probability_and_zero_value(self):
        q=torch.zeros(1,1,1,2)
        k=torch.zeros(1,1,2,2)
        v=torch.tensor([[[[2.,4.],[4.,2.]]]])
        plain=sink_attention(q,k,v)
        sink=sink_attention(q,k,v,sinks=torch.zeros(1,1,1,1))
        torch.testing.assert_close(plain.flatten(),torch.tensor([3.,3.]))
        torch.testing.assert_close(sink.flatten(),torch.tensor([2.,2.]))

    def test_all_masked_keys_sink_returns_zero_not_nan(self):
        q=torch.randn(1,1,1,2);k=torch.randn(1,1,2,2);v=torch.randn(1,1,2,2)
        mask=torch.full((1,1,1,2),float('-inf'))
        got=sink_attention(q,k,v,attention_mask=mask,sinks=torch.zeros(1,1,1,1))
        torch.testing.assert_close(got,torch.zeros_like(got))
        with self.assertRaises(ValueError):
            sink_attention(q,k,v,attention_mask=mask)

    def test_gqa_attention_bias_and_sink_vs_independent_scalar(self):
        q=torch.randn(1,4,3,4);k=torch.randn(1,2,5,4);v=torch.randn(1,2,5,4)
        bias=torch.randn(1,4,3,5)*.3
        mask=causal_additive_mask(torch.tensor([[2,3,4]]),torch.arange(5)[None])
        sinks=torch.randn(1,4,3,1)*.2
        got=sink_attention(q,k,v,attention_mask=mask,position_bias=bias,sinks=sinks)
        want=manual_attention(q,k,v,bias=bias,mask=mask,sinks=sinks)
        torch.testing.assert_close(got.double(),want,rtol=1e-5,atol=2e-6)

    def test_attention_invalid_shapes_rejected(self):
        q=torch.randn(1,3,2,4); k=torch.randn(1,2,2,4); v=torch.randn_like(k)
        with self.assertRaises(ValueError): sink_attention(q,k,v)
        with self.assertRaises(ValueError): sink_attention(torch.randn(1,2,2,4),k,v,sinks=torch.zeros(1,2,2))

    def test_gated_output_known(self):
        attn=torch.tensor([[[2.,4.]]]); h=torch.tensor([[[1.,0.]]])
        gate=torch.tensor([[0.,0.],[0.,0.]])
        out=torch.eye(2)
        torch.testing.assert_close(gated_output(attn,h,gate,out),attn*.5)

    def test_static_conv_scalar_golden_no_mask(self):
        x=torch.tensor([[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]])
        w=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        got=static_short_convolution(x,w)
        want=manual_static(x,w)
        torch.testing.assert_close(got.double(),want,rtol=1e-5,atol=1e-5)

    def test_static_conv_mask_keeps_residual(self):
        x=torch.tensor([[[1.],[5.],[3.]]])
        w=torch.tensor([[0.,1.]])
        mask=torch.tensor([[1,0,1]])
        got=static_short_convolution(x,w,padding_mask=mask)
        want=manual_static(x,w,mask)
        torch.testing.assert_close(got.double(),want,rtol=1e-6,atol=1e-6)
        self.assertGreater(float(got[0,1,0]),0) # masked source, residual persists

    def test_static_conv_decode_step_matches_full_prefill(self):
        x=torch.randn(2,11,4)
        w=torch.randn(4,4)*.25
        m=torch.tensor([[1,1,0,1,1,1,1,1,1,1,1],[1]*11])
        pre=static_short_convolution(x,w,padding_mask=m)
        hist=torch.zeros(2,3,4)
        steps=[]
        for t in range(x.shape[1]):
            y,hist=static_short_convolution_step(x[:,t:t+1],w,hist,padding_mask=m[:,t:t+1])
            steps.append(y)
        torch.testing.assert_close(torch.cat(steps,dim=1),pre,rtol=1e-5,atol=1e-5)

    def test_dynamic_conv_matches_independent_loop(self):
        x=torch.randn(2,5,4)
        d=torch.randn(2,5,3)
        s=torch.randn(3,4)*.1
        p=torch.randn(2*3,3)*.15
        got=dynamic_short_convolution(x,d,s,p,num_heads=2)
        want=manual_dynamic(x,d,s,p,2)
        torch.testing.assert_close(got.double(),want,rtol=1e-5,atol=1e-5)

    def test_dynamic_conv_seq_ids_and_mask(self):
        x=torch.randn(1,6,4);d=torch.randn(1,6,3)
        s=torch.randn(4,4)*.15; p=torch.randn(2*4,3)*.1
        mask=torch.tensor([[1,1,1,1,0,1]])
        ids=torch.tensor([[0,0,0,1,1,1]])
        got=dynamic_short_convolution(x,d,s,p,num_heads=2,padding_mask=mask,sequence_ids=ids)
        want=manual_dynamic(x,d,s,p,2,mask,ids)
        torch.testing.assert_close(got.double(),want,rtol=1e-5,atol=2e-6)

    def test_dynamic_conv_residual_scaling_differs_from_static(self):
        x=torch.ones(1,2,2);d=x;s=torch.zeros(4,2);p=torch.zeros(4,2)
        dyn=dynamic_short_convolution(x,d,s,p,num_heads=1)
        sta=static_short_convolution(x,torch.zeros(2,4))
        torch.testing.assert_close(dyn,torch.full_like(x,1/3))
        torch.testing.assert_close(sta,torch.full_like(x,1/math.sqrt(5)))

    def test_block_attention_one_state_returns_normalized_input(self):
        h=torch.tensor([[[3.,4.]]]);blocks=torch.empty(1,1,0,2)
        got=block_attention_residual(blocks,h,torch.tensor([2.,1.]))
        self.assertEqual(len(got),1)
        torch.testing.assert_close(got[0],unweighted_rmsnorm(h))

    def test_block_attention_three_queries_are_independent(self):
        h=torch.tensor([[[2.,0.]]]);blocks=torch.tensor([[[[0.,3.],[1.,1.]]]])
        q=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
        got=block_attention_residual(blocks,h,q)
        self.assertEqual(len(got),3)
        states=unweighted_rmsnorm(torch.cat([blocks,h.unsqueeze(-2)],-2)).double()
        for i in range(3):
            logits=(states[0,0]*q[i].double()).sum(-1)/math.sqrt(2)
            prob=torch.exp(logits-logits.max());prob=prob/prob.sum()
            want=(prob[:,None]*states[0,0]).sum(0)
            torch.testing.assert_close(got[i][0,0].double(),want,rtol=1e-6,atol=1e-6)

    def test_cross_pass_uses_correct_loop_layer_query(self):
        h=torch.tensor([[[0.,2.]]]);blocks=torch.tensor([[[[2.,0.]]]])
        queries=torch.zeros(12,4,2)
        queries[3]=torch.tensor([[3.,0.],[-3.,0.],[0.,3.],[0.,-3.]])
        got=cross_pass_messages(blocks,h,queries,3)
        self.assertEqual(got.shape,(1,1,4,2))
        torch.testing.assert_close(got,torch.stack(block_attention_residual(blocks,h,queries[3]),-2))
        self.assertGreater(abs((got[0,0,0]-got[0,0,1])).sum().item(),0.1)
        with self.assertRaises(ValueError):cross_pass_messages(blocks,h,queries,12)

    def test_bfloat16_controller_and_norm_preserve_contract(self):
        # Real Inkling 2 Mini core controllers use BF16 down/forget/gain weights.
        torch.manual_seed(5112)
        h=torch.randn(1,2,8,dtype=torch.bfloat16)
        m=torch.randn(1,2,8,dtype=torch.bfloat16)
        down=(torch.randn(4,16)*.1).to(torch.bfloat16)
        forget=(torch.randn(8,4)*.1).to(torch.bfloat16)
        gain=(torch.randn(8,4)*.1).to(torch.bfloat16)
        got=controller_message_gate(h,m,down,forget,gain)
        self.assertEqual(got.dtype,torch.float32) # HF explicitly promotes gate results
        self.assertTrue(torch.isfinite(got).all())
        self.assertEqual(unweighted_rmsnorm(h).dtype,h.dtype)

    def test_controller_zero_weights_is_identity(self):
        h=torch.randn(2,3,8);m=torch.randn_like(h)
        torch.testing.assert_close(controller_message_gate(h,m,torch.zeros(4,16),torch.zeros(8,4),torch.zeros(8,4)),h)

    def test_controller_tanh_negative_gate_does_not_equal_sigmoid(self):
        h=torch.tensor([[1.,2.]]);m=torch.tensor([[5.,9.]])
        down=torch.tensor([[1.,0.,0.,0.]])
        forget=torch.tensor([[-1.],[-1.]])
        gain=torch.tensor([[1.],[1.]])
        got=controller_message_gate(h,m,down,forget,gain)
        rank=unweighted_rmsnorm(h)[0,0]
        f=math.tanh(-rank);g=math.tanh(rank)
        want=(1-f)*h+g*m
        torch.testing.assert_close(got,want)

    def test_error_contracts_and_dynamic_geometry(self):
        with self.assertRaises(ValueError):unweighted_rmsnorm(torch.ones(2),eps=0)
        with self.assertRaises(ValueError):weighted_rmsnorm(torch.ones(3),torch.ones(4))
        with self.assertRaises(ValueError):static_short_convolution(torch.ones(1,2,3),torch.ones(2,4))
        with self.assertRaises(ValueError):dynamic_short_convolution(torch.ones(1,2,4),torch.ones(1,2,3),torch.ones(4,4),torch.ones(7,3),num_heads=2)
        with self.assertRaises(ValueError):controller_message_gate(torch.ones(1,2),torch.ones(1,3),torch.ones(1,4),torch.ones(2,1),torch.ones(2,1))

    def test_original_inkling_attention_and_kv_conv_geometry(self):
        # Actual Inkling 2 Mini text dimensions: D=2304, Q=24, KV=8, head=128, Kconv=4.
        q=torch.randn(1,24,2,128)*.02
        k=torch.randn(1,8,4,128)*.02
        v=torch.randn(1,8,4,128)*.02
        mask=causal_additive_mask(torch.tensor([[2,3]]),torch.arange(4)[None])
        out=sink_attention(q,k,v,attention_mask=mask,sinks=torch.zeros(1,24,2,1))
        self.assertEqual(tuple(out.shape),(1,2,24,128))
        self.assertTrue(torch.isfinite(out).all())
        kv=torch.randn(1,2,1024)*.02
        dynamic=torch.randn(1,2,2304)*.02
        static=torch.randn(4,1024)*.02
        proj=torch.randn(8*4,2304)*.02
        result=dynamic_short_convolution(kv,dynamic,static,proj,num_heads=8)
        self.assertEqual(tuple(result.shape),(1,2,1024))
        self.assertTrue(torch.isfinite(result).all())

    def test_full_loop_callback_reset_and_message_calls(self):
        rows=[]
        for layer in range(6):rows.append(("prelude",layer,0,layer))
        for p in range(3):
            for layer in range(6,18):rows.append(("core",layer,p,layer if p==0 else 24+(p-1)*12+layer-6))
        for layer in range(18,24):rows.append(("postlude",layer,0,layer))
        schedule=[{"ordinal":i,"stage":stage,"physical_layer":layer,"core_pass":p,"cache_slot":slot}
                  for i,(stage,layer,p,slot) in enumerate(rows)]
        calls=[]; message_calls=[]
        def layer_callback(step,hidden,blocks,messages):
            calls.append((step["physical_layer"],step["core_pass"],hidden.clone(),blocks.shape[-2],messages is not None))
            return hidden + 1
        def message_callback(prev_blocks,prev_hidden,index):
            message_calls.append((index,prev_blocks.shape[-2],prev_hidden.item()))
            return torch.zeros(*prev_hidden.shape[:2],4,prev_hidden.shape[-1])
        output,blocks,trace=execute_loop_callbacks(torch.zeros(1,1,1),schedule,layer_callback,message_callback)
        # 6 prelude + 12 core + 12 core + 12 core + 6 postlude, with restarts.
        self.assertEqual(output.item(),24.)
        self.assertEqual(len(trace),48)
        self.assertEqual(len(message_calls),24)
        self.assertEqual(calls[6][2].item(),6.)
        self.assertEqual(calls[18][2].item(),6.)
        self.assertEqual(calls[30][2].item(),6.)
        self.assertTrue(all(n==1 for _,n,_ in message_calls))
        self.assertEqual(blocks.shape[-2],4)  # prelude + core 2 + postlude
        self.assertEqual(sorted(x.cache_slot for x in trace),list(range(48)))
        bad=[dict(x) for x in schedule];bad[18]["physical_layer"]=7
        with self.assertRaises(ValueError):
            execute_loop_callbacks(torch.zeros(1,1,1),bad,layer_callback,message_callback)

    def test_loop_cache_trace_contract(self):
        sched=[]
        for i in range(6):sched.append((i,0,i))
        for p in range(3):
            for i in range(6,18):sched.append((i,p,i if p==0 else 24+(p-1)*12+i-6))
        for i in range(18,24):sched.append((i,0,i))
        rows=[{"ordinal":idx,"physical_layer":l,"core_pass":p,"cache_slot":slot} for idx,(l,p,slot) in enumerate(sched)]
        trace=schedule_trace(rows,torch.ones(1,2,4))
        self.assertEqual(len(trace),48)
        self.assertEqual(sorted({x.cache_slot for x in trace}),list(range(48)))
        self.assertEqual([x.layer for x in trace].count(6),3)
        with self.assertRaises(ValueError):schedule_trace(rows[:-1],torch.ones(1,2,4))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA PyTorch device')
class TestStage5ACuda(unittest.TestCase):
    def test_numerics_cpu_cuda_attention_and_convolution(self):
        torch.manual_seed(5222)
        q=torch.randn(1,4,4,8); k=torch.randn(1,2,6,8); v=torch.randn(1,2,6,8)
        bias=torch.randn(1,4,4,6)*.1
        mask=causal_additive_mask(torch.tensor([[2,3,4,5]]),torch.arange(6)[None])
        sink=torch.randn(1,4,4,1)*.1
        cpu=sink_attention(q,k,v,attention_mask=mask,position_bias=bias,sinks=sink)
        gpu=sink_attention(q.cuda(),k.cuda(),v.cuda(),attention_mask=mask.cuda(),position_bias=bias.cuda(),sinks=sink.cuda())
        torch.testing.assert_close(gpu.cpu(),cpu,rtol=2e-5,atol=5e-6)
        x=torch.randn(1,10,8); d=torch.randn(1,10,6)
        stat=torch.randn(4,8)*.05; proj=torch.randn(2*4,6)*.05
        c=dynamic_short_convolution(x,d,stat,proj,num_heads=2)
        g=dynamic_short_convolution(x.cuda(),d.cuda(),stat.cuda(),proj.cuda(),num_heads=2)
        torch.testing.assert_close(g.cpu(),c,rtol=2e-5,atol=5e-6)


if __name__=='__main__':unittest.main()

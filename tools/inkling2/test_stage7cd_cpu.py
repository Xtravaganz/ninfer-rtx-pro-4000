"""Host tests of real compiled ATen operators, no GPU inference claim."""
import sys,math,torch
from pathlib import Path

def main():
    torch.ops.load_library(str(Path(sys.argv[1]).resolve()))
    torch.manual_seed(31); ops=torch.ops.inkling7cd
    h=torch.randn(1,1,2304);q=torch.randn(3,2304);m=torch.randn_like(h)
    def rms(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    for n in (0,1,4):
        b=torch.randn(1,1,n,2304);s=rms(torch.cat([b,h.unsqueeze(-2)],-2))
        ref=((q@s.transpose(-1,-2))/math.sqrt(2304)).softmax(-1)@s
        torch.testing.assert_close(ops.residual(b,h,q,1e-6),ref,atol=1e-5,rtol=1e-5)
    d=torch.randn(64,4608)*.01;f=torch.randn(2304,64)*.01;g=torch.randn_like(f)*.01
    c=torch.nn.functional.linear(torch.cat([rms(h),rms(m)],-1),d)
    ref=(1-torch.nn.functional.linear(c,f).tanh())*h+torch.nn.functional.linear(c,g).tanh()*m
    torch.testing.assert_close(ops.controller(h,m,d,f,g,1e-6),ref)
    head=torch.randn(100,2304);logits=30*(head@h.reshape(-1)/30).tanh();v,i=ops.greedy(head,h,30.)
    assert int(i)==int(logits.argmax());torch.testing.assert_close(v,logits.max())
    # Saturation produces equal logits; original scanner chooses smallest ID.
    w=torch.tensor([[1000.,0.],[2000.,0.],[0.,0.]])
    v,i=ops.greedy(w,torch.tensor([1.,0.]),30.);assert int(i)==0 and float(v)==30.
    w[2,0]=float('nan');v,i=ops.greedy(w,torch.tensor([1.,0.]),30.);assert math.isnan(float(v))
    print('STAGE7CD HOST: PASS (3 residual, controller, greedy, tie, nonfinite)')
if __name__=='__main__':main()

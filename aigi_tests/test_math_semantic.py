import math
import pytest
import torch
from torch import nn
from aigi.math_ops import *
from aigi.semantic import TextCrossAttention,TokenWeights,MapPredictor,QuantizerControl,LocalAdaLN
from aigi.config import Components

def test_ste_gradient():
    x=torch.tensor([.2,1.8,-2.2],requires_grad=True)
    y=ste_round(x);assert torch.equal(y,x.round())
    y.sum().backward();assert torch.equal(x.grad,torch.ones_like(x))

def test_norm_constant_and_range():
    assert torch.all(spatial_norm(torch.ones(2,3,4,4))==.5)
    y=spatial_norm(torch.randn(2,3,4,4))
    assert torch.all(y.amin((-1,-2))==0) and torch.all(y.amax((-1,-2))==1)

def test_importance():
    assert torch.allclose(importance(torch.tensor([0.,.6,.9,1.])),torch.tensor([.1,.424,.829,1.]))

@pytest.mark.parametrize('value',[-6.,0.,2.])
def test_flow_identity(value):
    t=flow_time(torch.tensor(value));z0=torch.randn(2,3,4,4);e=torch.randn_like(z0)
    assert torch.allclose(flow_to_epsilon((1-t)*z0+t*e,e-z0,t),e,atol=5e-7)
    assert torch.allclose(((1-t)/t).square(),torch.tensor(value).exp(),rtol=1e-5)

def test_quadrature_keeps_negative_explanation():
    nodes=torch.linspace(-6,1,8);ce=torch.ones(8,1,1,2,2)*2;ne=ce*.5
    c,e=integrate_difficulty(ce,ne,nodes)
    assert torch.all(c==7) and torch.all(e==-3.5)
    with pytest.raises(ValueError): integrate_difficulty(ce,ne,nodes.flip(0))

def test_quantizer_importance_direction():
    m=quantization_scale(torch.tensor([2.,2.,100.]),torch.tensor([.1,1.,.1]),torch.zeros(3))
    assert m[0]>m[1] and m.max()<=16 and m.min()>=1

def test_cadc_moment_snr():
    assert torch.allclose(local_snr(torch.ones(1,4,2,2),torch.ones(1,4,2,2)),torch.full((1,1,2,2),24.))

def test_time_monotonic():
    s=torch.tensor([[[[1.,10.]]]]);w=torch.ones_like(s)*.5
    assert time_field(s,w,0)[0,0,0,0]>time_field(s,w,0)[0,0,0,1]
    assert torch.all(time_field(s,torch.ones_like(w),0)<time_field(s,torch.zeros_like(w),0))

def test_guidance_reconstruction():
    a,b,c=torch.randn(3,2,4,4)
    assert torch.allclose(guided_velocity(a,b,c,1.,1.),c,atol=5e-7)

@pytest.mark.parametrize('budget',[0.,.1,10.,1000.])
def test_waterfill_conservation(budget):
    w=torch.rand(2,1,4,4)*.9+.1;c=torch.rand_like(w)*3
    y=waterfill_target(w,c,torch.tensor([budget,budget]))
    assert torch.all(y>=0)
    assert torch.allclose(y.flatten(1).sum(1),torch.full((2,),budget),atol=3e-4)

def test_weighted_loss_unit_channels():
    e=torch.ones(2,3,8,8);w=torch.rand(2,1,2,2)
    assert torch.allclose(weighted_mean(e,w),torch.tensor(1.))

def test_padding_tiny_native():
    x=torch.arange(6).float().reshape(1,3,1,2);y,hw=pad_image(x)
    assert hw==(1,2) and y.shape==(1,3,256,256)
    assert torch.equal(y[:,:,:1,:2],x)

def test_cross_attention_zero_init_and_bad_mask():
    m=TextCrossAttention(8,12,8);x=torch.randn(2,8,3,3);t=torch.randn(2,5,12);mask=torch.ones(2,5,dtype=torch.bool)
    assert torch.equal(m(x,t,mask),x)
    with pytest.raises(ValueError):m(x,t,mask*False)

def test_token_weight_initialization_mask_and_grad():
    m=TokenWeights(12);a=torch.rand(2,5,4,4);t=torch.randn(2,5,12);mask=torch.tensor([[1,1,0,0,0]]).repeat(2,1).bool()
    s,w=m(a,t,mask);assert torch.allclose(w[:,:2],torch.full((2,2),.5));assert w[:,2:].max()==0
    s.sum().backward();assert m.mlp[-1].weight.grad is not None

def test_map_predictor_and_quantizer_ablations():
    m=MapPredictor(8,12,16);y=torch.randn(2,8,4,4);t=torch.randn(2,5,12);mask=torch.ones(2,5,dtype=torch.bool)
    maps=m(y,t,mask,3);assert maps['s'].shape==(2,1,4,4) and maps['c'].min()>0
    q=QuantizerControl(8,16);f=Components(texture=False,importance=False,correction=False)
    assert torch.all(q(y,y,maps,3,f)==1)
    f.importance=True;assert torch.all(q(y,y,maps,3,f)>=1)

def test_local_adaln_identity_gradient_and_remove():
    class B(nn.Module):
        def __init__(self):
            super().__init__();self.norm1=nn.LayerNorm(8);self.norm2=nn.LayerNorm(8)
    blocks=nn.ModuleList([B(),B()]);m=LocalAdaLN(blocks);x=torch.randn(1,4,8)
    ref=blocks[0].norm1(x);hs=m.install(blocks,torch.full((1,1,2,2),.2),.5,3)
    out=blocks[0].norm1(x);assert torch.equal(ref,out)
    out.sum().backward();assert m.maps[0][-1].weight.grad is not None
    for h in hs:h.remove()
    assert not blocks[0].norm1._forward_hooks

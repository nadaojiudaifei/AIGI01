"""Interface tests use a TEST-ONLY symbol recorder, NOT real CompressAI rANS.

These tests validate causal encode/decode scheduling; they do not validate
CompressAI wheels, real rANS, Sana weights or GPU execution.
"""
import json
import sys
import types
import pytest
import torch
from torch import nn
from aigi.entropy import SemanticEntropy
from aigi.config import Config
from aigi.math_ops import ste_round
from aigi.model import AIGICodec

class Recorder:
    def encode_with_indexes(self,symbols,indexes,*rest):self.symbols=symbols
    def flush(self):return json.dumps(self.symbols).encode()
class Reader:
    def set_stream(self,stream):self.values=iter(json.loads(stream))
    def decode_stream(self,indexes,*rest):return [next(self.values) for _ in indexes]
class GC(nn.Module):
    def __init__(self):
        super().__init__();self.quantized_cdf=torch.tensor([[0,1,2]]);self.cdf_length=torch.tensor([3]);self.offset=torch.tensor([0])
    def lower_bound_scale(self,x):return x.clamp_min(.11)
class Double(nn.Module):
    def forward(self,x):return torch.cat([x,x],1)
class Zero(nn.Module):
    def forward(self,x):return torch.zeros_like(x[:,:4])
class FakeCodec(nn.Module):
    M=4
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.3));self.h_s=nn.Identity()
        self.adapter_in=nn.ModuleList([Double() for _ in range(4)])
        self.g_c=nn.Identity();self.adapter_out=nn.ModuleList([nn.Conv2d(8,8,1) for _ in range(4)])
        self.LRP=nn.ModuleList([Zero() for _ in range(4)])
        self.g_s=self.scale=self.aux=self.prompt=nn.Identity();self.gaussian_conditional=GC();self.masks={}
    def get_mask_four_parts(self,b,c,h,w,device):
        ms=[]
        for k in range(4):
            m=torch.zeros(b,c,h,w,device=device);m[:,k]=1;ms.append(m)
        return ms
    def forward_with_mask(self,y,sigma,mu,mask):
        return (ste_round(y-mu)+mu)*mask,torch.ones_like(y)*.5*mask
    def compress_group_with_mask(self,gc,y,sigma,mu,mask,symbols,indexes):
        v=(y-mu).round()[mask.bool()];symbols.extend(v.int().tolist());indexes.extend([0]*len(v))
        return ((y-mu).round()+mu)*mask
    def decompress_group_with_mask(self,gc,sigma,mu,mask,reader,*rest):
        n=int(mask.sum());r=torch.zeros_like(mu);r[mask.bool()]=torch.tensor(reader.decode_stream([0]*n),dtype=mu.dtype)
        return (r+mu)*mask

@pytest.fixture
def fake_ans(monkeypatch):
    ans=types.ModuleType('compressai.ans');ans.BufferedRansEncoder=Recorder;ans.RansDecoder=Reader
    monkeypatch.setitem(sys.modules,'compressai.ans',ans)

@pytest.mark.parametrize('method',['cadc_style','idea1','idea2','joint'])
def test_causal_loop_decodes_without_original_image(fake_ans,method):
    base=FakeCodec();cfg=Config(method=method,hidden=8);net=SemanticEntropy(base,cfg).eval()
    # Make text conditioning nonzero so the test checks its actual causal use.
    for attn in net.text_context:
        nn.init.normal_(attn.out.weight,std=.02)
    z=torch.randn(1,4,2,2);y=torch.randn_like(z);text=torch.randn(1,5,2304);mask=torch.ones(1,5,dtype=torch.bool)
    with torch.no_grad():
        encoded=net._loop(z,text,mask,y,operation='encode')
        decoded=net._loop(z,text,mask,operation='decode',stream=encoded['y_string'])
        forward=net._loop(z,text,mask,y,operation='forward')
    assert decoded['maps'] is None and decoded['m'] is None
    for key in ('qhat','mean','scale','res','prompt','snr'):
        assert torch.equal(encoded[key],decoded[key]),key
        assert torch.allclose(forward[key],decoded[key]),key

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(4);self.norm2=nn.LayerNorm(4)
class DiT(nn.Module):
    def __init__(self):
        super().__init__();self.transformer_blocks=nn.ModuleList([Block()]);self.config=types.SimpleNamespace(timestep_scale=1.)
        self.weight=nn.Parameter(torch.tensor(.1))
    def forward(self,x,encoder_hidden_states,encoder_attention_mask,timestep,return_dict):
        b,c,h,w=x.shape;a=x.flatten(2).transpose(1,2);a=self.transformer_blocks[0].norm1(a);a=self.transformer_blocks[0].norm2(a)
        return (a.transpose(1,2).reshape_as(x)*self.weight+encoder_hidden_states.mean()*0.01,)
class Prompt(nn.Module):
    def forward(self,x):return x.mean((1,2,3))[:,None,None].expand(-1,3,2304)
class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.codec=FakeCodec();self.DiT=DiT();self.prompter=Prompt();self.timesteps=torch.tensor([999])
        self.sched=types.SimpleNamespace(base_scheduler=types.SimpleNamespace(timesteps=torch.tensor([999]),sigmas=torch.tensor([1.])),step=lambda pred,std,t,sample,**kw:sample-.5*pred)
    def render(self,z):return z

@pytest.mark.parametrize('method,nfe',[('ditic',1),('cadc_style',1),('idea1',1),('idea2',3),('joint',3)])
def test_decoder_branch_nfe_and_gradients(method,nfe):
    cfg=Config(method=method,hidden=8);net=AIGICodec(cfg,'cpu',backbone=FakeBackbone())
    x=torch.randn(1,4,2,2);out=dict(mean=x,scale=torch.zeros_like(x),prompt=x,res=x*.01,qhat=x,snr=torch.ones(1,1,2,2)*12)
    text=torch.randn(1,5,2304);mask=torch.ones(1,5,dtype=torch.bool)
    net.eval();image,extra=net.decode_features(out,text,mask,text*0,mask)
    assert extra['nfe']==nfe and image.shape==x.shape
    assert not net.blocks[0].norm1._forward_hooks
    net.train();image,extra=net.decode_features(out,text,mask,text*0,mask)
    assert extra['nfe']==1
    image.square().mean().backward();assert net.backbone.DiT.weight.grad is not None

"""Opt-in diagnostic instrumentation; never used for reported baseline latency.

Nested rows are inclusive and exclusive. FLOPs count registered PyTorch formulas
only; every unregistered tensor operator is listed, never silently called zero.
"""
from collections import Counter, defaultdict
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class OperatorInventory(TorchDispatchMode):
    def __init__(self, registry):
        super().__init__(); self.registry=registry; self.unregistered=Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result=func(*args,**(kwargs or {}))
        packet=func._overloadpacket
        leaves,_=tree_flatten((args,kwargs or {},result))
        if packet not in self.registry and any(isinstance(x,torch.Tensor) and x.is_floating_point() for x in leaves):
            self.unregistered[str(packet)]+=1
        return result


class ModuleRecorder:
    def __init__(self, modules, counter, sync=lambda:None):
        self.modules=modules;self.counter=counter;self.sync=sync
        self.stack=[];self.rows=defaultdict(lambda:dict(calls=0,inclusive_ms=0.,exclusive_ms=0.,inclusive_flops=0,exclusive_flops=0))
        self.handles=[]
    def __enter__(self):
        for name,module in self.modules.items():
            def pre(mod,args,name=name):
                self.sync();self.stack.append([name,time.perf_counter(),self.counter.get_total_flops(),0.,0])
            def post(mod,args,result,name=name):
                self.sync();key,beg,flops,child_time,child_flops=self.stack.pop()
                if key!=name:raise RuntimeError('Instrumented module stack mismatch')
                duration=(time.perf_counter()-beg)*1000;delta=self.counter.get_total_flops()-flops
                row=self.rows[name];row['calls']+=1;row['inclusive_ms']+=duration
                row['exclusive_ms']+=max(0.,duration-child_time);row['inclusive_flops']+=delta
                row['exclusive_flops']+=delta-child_flops
                if self.stack:self.stack[-1][3]+=duration;self.stack[-1][4]+=delta
            self.handles += [module.register_forward_pre_hook(pre),module.register_forward_hook(post)]
        return self
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.stack.clear()


def observed_modules(model):
    paths=['backbone.vae','backbone.aux_codec','backbone.codec.g_a','backbone.codec.h_a','backbone.codec.h_s',
           'backbone.codec.g_c','backbone.codec.g_s','backbone.codec.scale','backbone.codec.prompt','backbone.codec.aux',
           'backbone.prompter','backbone.DiT','entropy.student','entropy.decoder_maps','entropy.control','rho']
    for i in range(4):
        paths += [f'backbone.codec.{p}.{i}' for p in ('adapter_in','adapter_out','LRP')]
        paths += [f'entropy.text_context.{i}']
    modules={}
    for name in paths:
        try:modules[name]=model.get_submodule(name)
        except AttributeError:continue
    # AutoencoderDC.encode/decode bypass top-level forward: observe encoder and
    # decoder submodules, including when wrapped by PEFT.
    vae=model.backbone.vae
    if hasattr(vae,'get_base_model'):vae=vae.get_base_model()
    for part in ('encoder','decoder'):
        if hasattr(vae,part):modules['backbone.vae.'+part]=getattr(vae,part)
    return modules


def instrument(call, modules, sync=lambda:None):
    from torch.utils.flop_counter import FlopCounterMode
    counter=FlopCounterMode(display=False)
    inventory=OperatorInventory(counter.flop_registry)
    with counter, inventory, ModuleRecorder(modules,counter,sync) as recorder:
        output=call()
    rows={name:dict(stats,parameters=sum(p.numel() for p in modules[name].parameters()),
                     trainable_parameters=sum(p.numel() for p in modules[name].parameters() if p.requires_grad))
          for name,stats in recorder.rows.items()}
    report={'modules':rows,'registered_flops':counter.get_total_flops(),
            'registered_operations':{str(k):int(v) for k,v in counter.get_flop_counts().get('Global',{}).items()},
            'unregistered_float_tensor_ops':dict(inventory.unregistered),
            'flops_complete':False,'flops_convention':'PyTorch registered formulas; matrix multiply-add counts 2; excludes unregistered operators and entropy-coder integer work',
            'timing':'Diagnostic synchronized hooks only. Use uninstrumented repetitions for actual latency. Nested inclusive rows must not be summed.'}
    return output,report

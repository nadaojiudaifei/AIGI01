"""Atomic full-state checkpoints; strict shape/CDF loading and resumable optimizer/RNG."""
from pathlib import Path
import os
import random
import numpy as np
import torch
from .config import Config, Components
from .utils import sha256_file


def cfg_from_dict(d):
    d = dict(d)
    return Config(**{k: v for k, v in d.items() if k != 'components'}, components=Components(**d['components'])).validate()


def rng_state():
    ns = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4])}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    a, b, c, d, e = state['numpy']
    np.random.set_state((a, np.asarray(b, dtype=np.uint32), c, d, e))


def load_weights(model, state, maps_only=False):
    if maps_only:
        maps = {k: v for k, v in state.items() if k.startswith(('entropy.student.', 'entropy.decoder_maps.'))}
        own = model.state_dict()
        expected = {k for k in own if k.startswith(('entropy.student.', 'entropy.decoder_maps.'))}
        if set(maps) != expected:
            raise ValueError('Incomplete/incompatible map predictor checkpoint')
        own.update(maps)
        model.load_state_dict(own, strict=True)
        return
    prefix = 'backbone.codec.'
    codec_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if codec_state:
        model.backbone.codec.load_state_dict(codec_state, strict=True)
    model.load_state_dict(state, strict=True)
    model.backbone.codec.masks.clear()


def read_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != 'aigi01-training-v1':
        raise ValueError('Not an AIGI01 training checkpoint; base DiT-IC weights belong in cfg.weights')
    return value


def save_training(path, model, cfg, step, optimizer, aux_optimizer, discriminator=None,
                  disc_optimizer=None, ema=None, rank_rng=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    state = {'format': 'aigi01-training-v1', 'config': cfg.to_dict(), 'step': step,
             'data_sha256': {k: sha256_file(getattr(cfg,k)) for k in ('train_manifest','val_manifest')
                             if Path(getattr(cfg,k)).is_file()},
             'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'aux_optimizer': aux_optimizer.state_dict() if aux_optimizer else None,
             'discriminator': discriminator.state_dict() if discriminator else None,
             'disc_optimizer': disc_optimizer.state_dict() if disc_optimizer else None,
             'ema': ema, 'rng': rank_rng or [rng_state()]}
    torch.save(state, tmp)
    os.replace(tmp, p)


class EMA:
    def __init__(self, model, decay=.999):
        self.decay = decay
        # CPU shadow only for trainable params, not the frozen ~1B backbone.
        self.shadow = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach().cpu(), 1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        parameters = dict(model.named_parameters())
        for name, value in self.shadow.items():
            parameters[name].copy_(value.to(parameters[name]))


def compare_training(first, second, output):
    from .utils import atomic_json
    a,b=read_checkpoint(first),read_checkpoint(second)
    def equal(x,y,path):
        if torch.is_tensor(x):
            if not torch.is_tensor(y) or x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x,y):
                raise ValueError('Training states differ: '+path)
        elif isinstance(x,dict):
            if not isinstance(y,dict) or set(x)!=set(y):raise ValueError('Training dictionary differs: '+path)
            for key in x:equal(x[key],y[key],path+'.'+str(key))
        elif isinstance(x,(list,tuple)):
            if not isinstance(y,type(x)) or len(x)!=len(y):raise ValueError('Training sequence differs: '+path)
            for i,(xx,yy) in enumerate(zip(x,y)):equal(xx,yy,path+'.'+str(i))
        elif x!=y:raise ValueError('Training value differs: '+path)
    for key in ('model','optimizer','aux_optimizer','discriminator','disc_optimizer','ema','step','rng','data_sha256'):
        equal(a.get(key),b.get(key),key)
    result={'passed':True,'comparison':'exact tensors, optimizers, auxiliary/discriminator/EMA, RNG, step and data hashes',
            'first_sha256':sha256_file(first),'second_sha256':sha256_file(second)}
    atomic_json(result,output);return result

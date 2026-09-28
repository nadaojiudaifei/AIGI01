"""Frozen Sana teacher: actual cross-attention and paired flow-noise error quadrature."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
from .math_ops import spatial_norm, flow_time, flow_to_epsilon, integrate_difficulty
from .utils import sha256_file, seed_all, atomic_json
from .data import ManifestDataset, cache_key
from .text import TextEncoder


class AttentionCapture:
    def __init__(self, transformer, mask, layers=(3, 10, 17, 24)):
        self.mask, self.layers, self.maps, self.handles = mask, layers, [], []
        for index in layers:
            block = transformer.transformer_blocks[index]
            self.handles.append(block.attn2.register_forward_pre_hook(self.capture, with_kwargs=True))

    def capture(self, attn, args, kwargs):
        h = kwargs.get('hidden_states', args[0] if args else None)
        text = kwargs.get('encoder_hidden_states', args[1] if len(args) > 1 else None)
        if h is None or text is None:
            raise RuntimeError('Sana attn2 API incompatible with the pinned capture adapter')
        q, k = attn.to_q(h), attn.to_k(text)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        batch, heads = q.shape[0], attn.heads
        dim = q.shape[-1] // heads
        q = q.reshape(batch, -1, heads, dim).transpose(1, 2).float()
        k = k.reshape(batch, -1, heads, dim).transpose(1, 2).float()
        scores = (q @ k.transpose(-1, -2)) / dim ** .5
        scores = scores.masked_fill(~self.mask[:, None, None, :].bool(), -1e4)
        self.maps.append(scores.softmax(-1).mean(1).detach())

    def consume(self, height, width):
        if len(self.maps) != len(self.layers):
            raise RuntimeError('Teacher did not execute every requested cross-attention layer')
        a = torch.stack(self.maps).mean(0).transpose(1, 2)
        self.maps.clear()
        return a.reshape(a.shape[0], a.shape[1], height, width)

    def close(self):
        for h in self.handles:
            h.remove()


class Teacher:
    def __init__(self, cfg, device='cuda'):
        from diffusers import AutoencoderDC, SanaTransformer2DModel
        kwargs = dict(local_files_only=cfg.offline, torch_dtype=torch.float32)
        if cfg.model_variant:
            kwargs['variant'] = cfg.model_variant
        self.vae = AutoencoderDC.from_pretrained(cfg.sana_path, subfolder='vae', **kwargs).to(device).eval().requires_grad_(False)
        self.transformer = SanaTransformer2DModel.from_pretrained(cfg.sana_path, subfolder='transformer', **kwargs).to(device).eval().requires_grad_(False)
        if len(self.transformer.transformer_blocks) != 28:
            raise ValueError('Teacher adapter requires Sana-600M with 28 blocks')
        self.text = TextEncoder(cfg.sana_path, cfg.offline, cfg.text_length)
        self.cfg, self.device = cfg, device
        digest = hashlib.sha256(self.text.fingerprint.encode())
        for folder in ('transformer', 'vae'):
            for p in sorted((Path(cfg.sana_path) / folder).rglob('*')):
                if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin'}:
                    digest.update(bytes.fromhex(sha256_file(p)))
        self.fingerprint = digest.hexdigest()

    @torch.no_grad()
    def compute(self, x, prompt, sample_seed):
        cfg, device = self.cfg, self.device
        text, mask, null, nmask = self.text.paired([prompt], device)
        z0 = self.vae.encode(x.to(device)).latent * self.vae.config.scaling_factor
        lo, hi = -6., [-1., 0., 1., 2.][cfg.quality - 1]
        nodes = torch.linspace(lo, hi, 8, device=device)
        cond_errors, null_errors, attentions = [], [], []
        rng = torch.Generator(device=device).manual_seed(sample_seed)
        for log_snr in nodes:
            ce, ne, am = [], [], []
            t = flow_time(log_snr)
            for _ in range(2):
                noise = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=rng)
                zt = (1 - t) * z0 + t * noise
                timestep = (1000. * t).expand(z0.shape[0]) * getattr(self.transformer.config, 'timestep_scale', 1.)
                capture = AttentionCapture(self.transformer, mask)
                try:
                    vp = self.transformer(zt, encoder_hidden_states=text, encoder_attention_mask=mask,
                                          timestep=timestep, return_dict=False)[0]
                    am.append(capture.consume(*zt.shape[-2:]))
                finally:
                    capture.close()
                vn = self.transformer(zt, encoder_hidden_states=null, encoder_attention_mask=nmask,
                                      timestep=timestep, return_dict=False)[0]
                ce.append((noise - flow_to_epsilon(zt, vp, t)).square().mean(1, keepdim=True))
                ne.append((noise - flow_to_epsilon(zt, vn, t)).square().mean(1, keepdim=True))
            cond_errors.append(torch.stack(ce).mean(0))
            null_errors.append(torch.stack(ne).mean(0))
            attentions.append(torch.stack(am).mean(0))
        c, e = integrate_difficulty(torch.stack(cond_errors), torch.stack(null_errors), nodes)
        attention = torch.stack(attentions).mean(0)
        uniform = mask.float() / mask.sum(1, keepdim=True)
        s = spatial_norm((spatial_norm(attention) * uniform[..., None, None]).sum(1, keepdim=True))
        return {k: v[0].detach().cpu().numpy() for k, v in
                dict(attention=attention, s=s, c=c, e=e, text=text, mask=mask, null_text=null, null_mask=nmask).items()}


def build_cache(cfg, manifest, device='cuda', limit=0):
    seed_all(cfg.seed, cfg.deterministic)
    data = ManifestDataset(manifest, cfg.crop, augmentation='center' if Path(manifest).resolve() == Path(cfg.val_manifest).resolve() else cfg.augmentation,
                           views_per_image=1 if Path(manifest).resolve() == Path(cfg.val_manifest).resolve() else cfg.views_per_image, seed=cfg.seed)
    teacher = Teacher(cfg, device)
    out = Path(cfg.teacher_cache)
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(version='teacher-v1', teacher_sha256=teacher.fingerprint, text_sha256=teacher.text.fingerprint)
    contract_path = out / 'cache_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Cache directory belongs to different teacher/text assets')
    atomic_json(contract, contract_path)
    for i in range(len(data) if not limit else min(limit, len(data))):
        row = data.row_at(i)
        key = data.key_at(i)
        target = out / (key + '.npz')
        if target.is_file():
            with np.load(target, allow_pickle=False) as f:
                metadata = json.loads(str(f['metadata']))
            if metadata.get('teacher_sha256') != teacher.fingerprint:
                raise ValueError('Existing cache was produced by different teacher assets; use a separate cache directory')
            continue
        item = data[i]
        arrays = teacher.compute(item['image'].unsqueeze(0), item['prompt'], int(key[:12], 16) % 2**31)
        metadata = dict(cache_key=key, teacher_sha256=teacher.fingerprint, text_sha256=teacher.text.fingerprint,
                        view=data.spec_at(i), quadrature_nodes=8, noise_draws=2, crop=cfg.crop, quality=cfg.quality,
                        prompt_origin=row.get('prompt_origin', 'unknown'), version='teacher-v1')
        tmp = target.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, metadata=np.array(json.dumps(metadata)), **arrays)
        os.replace(tmp, target)
        print(json.dumps({'teacher_sample': i + 1, 'total': len(data), 'cache': str(target)}), flush=True)

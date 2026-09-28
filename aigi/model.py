"""Image -> entropy bitstream -> standalone decoder, atop original DiT-IC parts."""
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from .entropy import SemanticEntropy
from .semantic import LocalAdaLN
from .math_ops import time_field, guided_velocity, guidance_weights


@contextmanager
def coding_determinism():
    old = torch.backends.cudnn.deterministic
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = old
        torch.backends.cudnn.benchmark = bench


class AIGICodec(nn.Module):
    def __init__(self, cfg, device='cuda', training=False, backbone=None):
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            from .backbone import Backbone
            backbone = Backbone(cfg, device, training)
        self.backbone = backbone
        self.entropy = None if cfg.method == 'ditic' else SemanticEntropy(backbone.codec, cfg)
        self.rho = self.adaln = None
        if cfg.method in {'idea2','joint'}:
            self.rho = nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,1))
            nn.init.zeros_(self.rho[-1].weight)
            nn.init.zeros_(self.rho[-1].bias)
            self.adaln = LocalAdaLN(self.blocks)
        self.to(device)
        self.train(training)

    @property
    def blocks(self):
        dit = self.backbone.DiT
        if hasattr(dit, 'get_base_model'):
            dit = dit.get_base_model()
        return dit.transformer_blocks

    def update(self):
        self.backbone.codec.update(force=True)

    def _predict(self, sample, text, mask, tmap=None):
        b = self.backbone
        timestep = b.sched.base_scheduler.timesteps[:1].to(sample.device).expand(sample.shape[0])
        if tmap is not None:
            timestep = tmap.mean((1, 2, 3)) * 1000.
        timestep = timestep * getattr(b.DiT.config, 'timestep_scale', 1.)
        handles = []
        try:
            if tmap is not None and self.cfg.components.local_adaln:
                handles = self.adaln.install(self.blocks, tmap, self.cfg.beta, self.cfg.quality)
            return b.DiT(sample, encoder_hidden_states=text, encoder_attention_mask=mask,
                         timestep=timestep, return_dict=False)[0]
        finally:
            for handle in handles:
                handle.remove()

    def _conditions(self, out, text, mask, null_text, null_mask, use_y=True):
        latent = self.backbone.prompter(out['prompt'])
        if not self.cfg.components.latent_prompt:
            use_y = False
        if use_y:
            return torch.cat([text, latent], 1), torch.cat([mask, torch.ones(latent.shape[:2], device=mask.device, dtype=torch.bool)], 1)
        return text, mask

    def decode_features(self, out, text, mask, null_text, null_mask):
        b, flags = self.backbone, self.cfg.components
        sample = out['mean']
        if self.cfg.method in {'ditic', 'cadc_style'}:
            pos = b.prompter(out['prompt'])
            if not flags.latent_prompt:
                pos = torch.zeros_like(pos)
            pred = self._predict(sample, pos, None)
            std = torch.exp(.5 * out['scale'].clamp(-30, 20))
            if flags.variance_flow:
                z = b.sched.step(pred, std, b.timesteps, sample, return_dict=True) + out['res']
            else:
                z = sample - b.sched.base_scheduler.sigmas[0].to(sample.device) * pred + out['res']
            return b.render(z), {'z': z, 'nfe': 1}
        pos, pmask = self._conditions(out, text, mask, null_text, null_mask)
        idea2 = self.cfg.method in {'idea2', 'joint'}
        aux = {}
        if not idea2:
            pred = self._predict(sample, pos, pmask)
            std = torch.exp(.5 * out['scale'].clamp(-30, 20))
            if flags.variance_flow:
                z = b.sched.step(pred, std, b.timesteps, sample, return_dict=True) + out['res']
            else:
                z = sample - b.sched.base_scheduler.sigmas[0].to(sample.device) * pred + out['res']
            return b.render(z), {'z': z, 'nfe': 1}
        maps = self.entropy.decoder_maps(out['qhat'], text, mask, self.cfg.quality, flags.learned_token_weights)
        snr = F.interpolate(out['snr'], sample.shape[-2:], mode='bilinear', align_corners=False)
        w = F.interpolate(maps['w'], sample.shape[-2:], mode='bilinear', align_corners=False)
        if not flags.snr_time:
            # Global-SNR ablation, preserving a non-arbitrary reference rate.
            snr = snr.mean((2, 3), keepdim=True).expand_as(snr)
        rho = .25 * self.rho(torch.cat([torch.log1p(snr), w,
                                       torch.full_like(w, self.cfg.beta),
                                       torch.full_like(w, self.cfg.quality / 4.)], 1)).tanh()
        if not flags.time_residual:
            rho = torch.zeros_like(rho)
        t = time_field(snr, w, rho, self.cfg.beta if flags.semantic_time else 0.)
        if self.training:
            # Independent 10% condition drops train all branches; no empty attention masks.
            drop_p = torch.rand((sample.shape[0], 1, 1), device=sample.device) < .1
            train_text = torch.where(drop_p, null_text, text)
            train_mask = torch.where(drop_p.squeeze(-1), null_mask, mask)
            use_y = bool(torch.rand((), device=sample.device) >= .1)
            p, m = self._conditions(out, train_text, train_mask, null_text, null_mask, use_y)
            pred = self._predict(sample, p, m, t)
            nfe = 1
        elif flags.dual_guidance:
            v00 = self._predict(sample, null_text, null_mask, t)
            vp0 = self._predict(sample, text, mask, t)
            vpy = self._predict(sample, pos, pmask, t)
            zp, zy = guidance_weights(t, w, snr)
            pred = guided_velocity(v00, vp0, vpy, zp, zy)
            nfe = 3
        else:
            pred = self._predict(sample, pos, pmask, t)
            nfe = 1
        z = sample - t * pred + out['res']
        return b.render(z), {'z': z, 't': t, 'decoder_maps': maps, 'nfe': nfe}

    def forward(self, x, text, mask, null_text, null_mask):
        latent, aux = self.backbone.analyze_image(x)
        if self.cfg.stage == 'maps':
            with torch.no_grad():
                y = self.backbone.codec.g_a(latent, aux)
                decoded = self.entropy(latent, aux, text, mask)['qhat']
            f = self.cfg.components
            return {'maps': self.entropy.student(y, text, mask, self.cfg.quality, f.learned_token_weights),
                    'decoder_maps': self.entropy.decoder_maps(decoded, text, mask, self.cfg.quality, f.learned_token_weights)}
        if self.cfg.method == 'ditic':
            out = self.backbone.codec(latent, aux)
        else:
            out = self.entropy(latent, aux, text, mask)
        image, more = self.decode_features(out, text, mask, null_text, null_mask)
        return {'image': image, 'latent': latent, **out, **more}

    @contextmanager
    def entropy_on_cpu(self):
        # Decode CDF/means in the same FP32 CPU reference environment at both endpoints.
        # Image analysis/generation stay on GPU; transfer costs are included in timings.
        device = next(self.backbone.DiT.parameters()).device
        threads = torch.get_num_threads()
        self.backbone.codec.to('cpu')
        self.backbone.codec.masks.clear()
        if self.entropy is not None:self.entropy.to('cpu')
        torch.set_num_threads(1)
        try:
            yield
        finally:
            self.backbone.codec.to(device)
            self.backbone.codec.masks.clear()
            if self.entropy is not None:self.entropy.to(device)
            torch.set_num_threads(threads)

    @torch.no_grad()
    def compress(self, x, text, mask):
        if self.training:
            raise RuntimeError('Call eval() before real compression')
        if x.shape[0] != 1:
            raise ValueError('One image per bitstream')
        with coding_determinism():
            latent, aux = self.backbone.analyze_image(x)
            with self.entropy_on_cpu():
                if self.cfg.method == 'ditic':
                    return self.backbone.codec.compress(latent.cpu(), aux.cpu())
                return self.entropy.compress(latent.cpu(), aux.cpu(), text.cpu(), mask.cpu())

    @torch.no_grad()
    def decompress(self, strings, shape, text, mask, null_text, null_mask):
        if self.training:
            raise RuntimeError('Call eval() before decompression')
        with coding_determinism():
            with self.entropy_on_cpu():
                if self.cfg.method == 'ditic':
                    mean, scale, res, prompt = self.backbone.codec.decompress(strings, shape)
                    out = dict(mean=mean, scale=scale, res=res, prompt=prompt)
                else:
                    out = self.entropy.decompress(strings, shape, text.cpu(), mask.cpu())
            device = next(self.backbone.DiT.parameters()).device
            out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}
            return self.decode_features(out, text.to(device), mask.to(device), null_text.to(device), null_mask.to(device))

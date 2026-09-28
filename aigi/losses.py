"""Explicit normalized RD, perceptual, map-distillation and water-filling objectives."""
import math
import zlib
import torch
from torch import nn
import torch.nn.functional as F
from .math_ops import weighted_mean, waterfill_target


def map_loss(pred, batch, flags):
    terms = []
    # Transforms prevent a large finite-quadrature energy from overwhelming importance.
    for key in ('s', 'c', 'e'):
        if key == 'c' and not flags.difficulty:
            continue
        if key == 'e' and not flags.explanation:
            continue
        if key == 's' and not flags.importance:
            continue
        p = pred[key]
        target = F.interpolate(batch[key], p.shape[-2:], mode='bilinear', align_corners=False)
        if key == 'c':
            p, target = torch.log1p(p), torch.log1p(target.clamp_min(0))
        elif key == 'e':
            p, target = torch.asinh(p), torch.asinh(target)
        terms.append(F.l1_loss(p, target))
    if not terms:
        return pred['s'].sum() * 0
    # Weak prior prevents learned token scores from collapsing to a single token.
    omega = pred['omega'].clamp_min(1e-8)
    valid = batch['mask'].float()
    kl = (omega * (omega.log() + valid.sum(1, keepdim=True).log()) * valid).sum(1).mean()
    return sum(terms) + .001 * kl


class RateDistortionLoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.lpips = self.dists = self.discriminator = None
        if cfg.stage == 'codec':
            if cfg.lpips_weight:
                import lpips
                self.lpips = lpips.LPIPS(net='alex', spatial=True).to(device).eval().requires_grad_(False)
            if cfg.dists_weight:
                import pyiqa
                self.dists = pyiqa.create_metric('dists', as_loss=True, device=device).eval().requires_grad_(False)
            if cfg.gan_weight:
                from taming.modules.losses.vqperceptual import NLayerDiscriminator, weights_init
                self.discriminator = NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=False).apply(weights_init).to(device)

    def forward(self, out, batch, step):
        cfg, flags = self.cfg, self.cfg.components
        zero = batch['image'].sum() * 0
        maps = zero
        if flags.map_distillation and cfg.method in {'idea1', 'idea2', 'joint'}:
            if out.get('maps') is not None:
                maps = maps + map_loss(out['maps'], batch, flags)
            if out.get('decoder_maps') is not None:
                maps = maps + map_loss(out['decoder_maps'], batch, flags)
        if cfg.stage == 'maps':
            return maps, {'map_loss': maps.detach()}
        original, recon = batch['image'], out['image']
        if flags.weighted_loss and cfg.method in {'idea1', 'joint'}:
            w = out['maps']['w'].detach()
        else:
            w = torch.ones_like(original[:, :1])
        pixel = weighted_mean((original - recon).square(), w)
        lpips_loss = weighted_mean(self.lpips(original, recon), w) if self.lpips is not None else zero
        # DISTS is global. Its inputs are [0,1]; no fabricated spatial DISTS map.
        dists_loss = self.dists((recon + 1) / 2, (original + 1) / 2).mean() if self.dists is not None else zero
        likelihoods = out['likelihoods']
        pixels = original.shape[0] * original.shape[-2] * original.shape[-1]
        ry = -torch.log2(likelihoods['y'].clamp_min(1e-9)).sum() / pixels
        rz = -torch.log2(likelihoods['z'].clamp_min(1e-9)).sum() / pixels
        # Actual compressed prompt bytes are a constant w.r.t. image-codec weights.
        prompt_bits = 0 if cfg.method in {'ditic', 'cadc_style'} else sum(8 * len(zlib.compress(p.encode(), 9)) for p in batch['prompt'])
        rprompt = zero + prompt_bits / pixels
        wf = zero
        if flags.waterfill and cfg.method in {'idea1', 'joint'}:
            r = -torch.log2(likelihoods['y'].clamp_min(1e-9)).sum(1, keepdim=True)
            ww = out['maps']['w'].detach() if flags.importance else torch.ones_like(r)
            cc = out['maps']['c'] if flags.difficulty else torch.zeros_like(r)
            target = waterfill_target(ww, cc, r.detach().flatten(1).sum(1))
            wf = (r - target).abs().mean() / likelihoods['y'].shape[1]
        sd = zero
        if flags.self_distillation:
            sd = .05 * F.relu(.5 - F.cosine_similarity(out['z'], out['latent'], dim=1)).mean()
        nll = pixel + cfg.lpips_weight * lpips_loss + cfg.dists_weight * dists_loss
        anneal = min(1., (step + 1) / max(1, .1 * cfg.steps))
        self.reconstruction_objective = nll
        total = nll + cfg.rate_weight * (ry + rz + rprompt) + .1 * sd + cfg.distill_weight * maps + cfg.waterfill_weight * anneal * wf
        return total, {'loss': total.detach(), 'pixel': pixel.detach(), 'lpips': lpips_loss.detach(),
                       'dists': dists_loss.detach(), 'bpp_y_est': ry.detach(), 'bpp_z_est': rz.detach(),
                       'bpp_prompt': rprompt.detach(), 'map_loss': maps.detach(),
                       'waterfill': wf.detach(), 'self_distillation': sd.detach()}

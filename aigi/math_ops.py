"""Pure differentiable definitions shared by training and the real decoder."""
import math
import torch
import torch.nn.functional as F


def ste_round(x):
    return x + (x.round() - x).detach()


def spatial_norm(x, eps=1e-6):
    lo, hi = x.amin((-2, -1), keepdim=True), x.amax((-2, -1), keepdim=True)
    # Constant attention is non-informative, not a claim that the region is unimportant.
    return torch.where(hi - lo > eps, (x - lo) / (hi - lo).clamp_min(eps), torch.full_like(x, .5))


def importance(s, w_min=.1, eta=2.):
    return w_min + (1. - w_min) * s.clamp(0, 1).pow(eta)


def flow_time(log_snr):
    return torch.sigmoid(-.5 * log_snr)


def flow_to_epsilon(z_t, velocity, t):
    return z_t + (1 - t) * velocity


def integrate_difficulty(cond_errors, null_errors, log_snr):
    """Errors: [K,B,1,H,W], already averaged across noise draws and channels."""
    if len(log_snr) < 2 or not bool(torch.all(log_snr[1:] > log_snr[:-1])):
        raise ValueError('log-SNR quadrature nodes must be strictly increasing')
    c = .5 * torch.trapezoid(cond_errors, log_snr, dim=0)
    e = .5 * torch.trapezoid(null_errors - cond_errors, log_snr, dim=0)
    return c.clamp_min(0), e  # e is signed; do not hide adverse prompt effects.


def quantization_scale(texture, w, delta, maximum=16.):
    return (texture.clamp_min(1).log() - .5 * w.clamp_min(1e-6).log() + delta).clamp(0, math.log(maximum)).exp()


def local_snr(mu, sigma, delta=1.):
    # Moments are those of entropy-coded symbols, BEFORE synthesis or LRP.
    return (mu.square() + sigma.square()).mean(1, keepdim=True) / (delta * delta / 12.)


def time_field(snr, w, rho, beta=.5, t_min=.02, t_max=.98):
    logits = -.5 * snr.clamp_min(1e-8).log() - beta * (2 * w - 1) + rho
    return logits.sigmoid().clamp(t_min, t_max)


def guidance_weights(t, w, snr):
    return 1. + t * (1. - w), 1. + w * snr / (1. + snr)


def guided_velocity(v00, vp0, vpy, zp, zy):
    return v00 + zp * (vp0 - v00) + zy * (vpy - vp0)


def waterfill_target(w, c, total_bits, iterations=48):
    """Per image target, channel-summed bits per spatial cell, non-differentiable target."""
    with torch.no_grad():
        score = .5 * torch.log2(w.clamp_min(1e-6)) + c / math.log(2)
        flat = score.flatten(1)
        budget = total_bits.reshape(-1, 1).clamp_min(0)
        lo = flat.min(1, keepdim=True).values - budget - 1
        hi = flat.max(1, keepdim=True).values + 1
        for _ in range(iterations):
            mid = (lo + hi) * .5
            over = (flat - mid).clamp_min(0).sum(1, keepdim=True) > budget
            lo, hi = torch.where(over, mid, lo), torch.where(over, hi, mid)
        return (flat - (lo + hi) * .5).clamp_min(0).reshape_as(w)


def weighted_mean(error, w):
    w = F.interpolate(w, error.shape[-2:], mode='bilinear', align_corners=False)
    return ((error * w).flatten(1).sum(1) / (w.flatten(1).sum(1) * error.shape[1]).clamp_min(1e-8)).mean()


def pad_image(x, multiple=256):
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % multiple, 0, (-h) % multiple), mode='replicate'), (h, w)

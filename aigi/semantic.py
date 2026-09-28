"""Learned components; no face/OCR priors and no user token-weight overrides."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .math_ops import importance, spatial_norm, quantization_scale


class TextCrossAttention(nn.Module):
    def __init__(self, channels, text_dim=2304, hidden=128):
        super().__init__()
        self.q = nn.Conv2d(channels, hidden, 1)
        self.k = nn.Linear(text_dim, hidden)
        self.v = nn.Linear(text_dim, hidden)
        self.out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, text, mask):
        if not bool(mask.any(1).all()):
            raise ValueError('Every text sequence must contain at least one valid token')
        q = self.q(x).flatten(2).transpose(1, 2)
        logits = q @ self.k(text).transpose(1, 2) / math.sqrt(q.shape[-1])
        a = logits.masked_fill(~mask[:, None, :].bool(), -1e4).softmax(-1)
        h = (a @ self.v(text)).transpose(1, 2).reshape(x.shape[0], -1, *x.shape[-2:])
        return x + self.out(h)


class MapPredictor(nn.Module):
    def __init__(self, channels=320, text_dim=2304, hidden=128):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU())
        self.attn = TextCrossAttention(hidden, text_dim, hidden)
        self.qmap = nn.Conv2d(hidden, 64, 1)
        self.kmap = nn.Linear(text_dim, 64)
        self.token_weights = TokenWeights(text_dim)
        self.net = nn.Sequential(nn.Conv2d(hidden + 1, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, 2, 1))

    def forward(self, y, text, mask, quality, learned_weights=True):
        h = self.attn(self.pre(y), text, mask)
        qmap = self.qmap(h).flatten(2).transpose(1, 2)
        logits = (qmap @ self.kmap(text).transpose(1, 2)) / 8.
        attention = logits.masked_fill(~mask[:, None, :].bool(), -1e4).softmax(-1)
        attention = attention.transpose(1, 2).reshape(y.shape[0], text.shape[1], *y.shape[-2:])
        s, omega = self.token_weights(attention, text, mask, learned_weights)
        q = torch.full_like(h[:, :1], quality / 4.)
        c, e = self.net(torch.cat([h, q], 1)).chunk(2, 1)
        return {'s': s, 'w': importance(s), 'c': F.softplus(c), 'e': e,
                'attention': attention, 'omega': omega}


class TokenWeights(nn.Module):
    def __init__(self, dim=2304):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, maps, text, mask, learned=True):
        # maps [B,K,H,W]. Learn token scores; initialize to uniform valid-token weights.
        a = spatial_norm(maps)
        logits = self.mlp(text).squeeze(-1) if learned else torch.zeros_like(mask, dtype=text.dtype)
        logits = logits.masked_fill(~mask.bool(), -1e4)
        omega = logits.softmax(-1)
        s = spatial_norm((a * omega[..., None, None]).sum(1, keepdim=True))
        return s, omega


class QuantizerControl(nn.Module):
    def __init__(self, channels=320, hidden=128):
        super().__init__()
        self.texture = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        self.delta = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, y, hyper, maps, quality, flags):
        tex = 1 + F.softplus(self.texture(y - hyper)) if flags.texture else torch.ones_like(y[:, :1])
        w = maps['w'] if flags.importance else torch.ones_like(tex)
        c = torch.log1p(maps['c']) if flags.difficulty else torch.zeros_like(tex)
        e = torch.asinh(maps['e']) if flags.explanation else torch.zeros_like(tex)
        s = maps['s'] if flags.importance else torch.zeros_like(tex)
        q = torch.full_like(tex, quality / 4.)
        delta = .5 * torch.tanh(self.delta(torch.cat([s, c, e, q], 1))) if flags.correction else torch.zeros_like(tex)
        return quantization_scale(tex, w, delta)


class LocalAdaLN(nn.Module):
    """Inject local affine modulation after norm1 / norm2 without editing upstream files.

    Original global Sana timestep conditioning remains; this adds a tokenwise residual.
    Hooks are scoped to a single forward; gradient checkpoint recomputation is disabled.
    """
    def __init__(self, blocks, hidden=64):
        super().__init__()
        self.maps = nn.ModuleList()
        for b in blocks:
            dim = b.norm1.normalized_shape[-1]
            net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.maps.append(net)

    def install(self, blocks, t, beta, quality):
        feat = torch.stack([torch.logit(t.clamp(.001, .999)).flatten(1),
                            torch.full_like(t.flatten(1), beta),
                            torch.full_like(t.flatten(1), quality / 4.)], -1)
        handles = []
        for block, net in zip(blocks, self.maps):
            scale, shift = net(feat).chunk(2, -1)
            def mod(_m, _args, out, s=scale, b=shift):
                if not torch.is_tensor(out) or out.shape != s.shape:
                    raise RuntimeError('Sana norm shape incompatible with local AdaLN')
                return (1 + .25 * torch.tanh(s)) * out + .25 * torch.tanh(b)
            handles.append(block.norm1.register_forward_hook(mod))
            handles.append(block.norm2.register_forward_hook(mod))
        return handles

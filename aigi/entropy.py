"""One causal four-group implementation for forward, real rANS encode and decode.

The original DiT-IC transforms, context blocks and entropy models are reused.
The encoder-only m/S/c/e are never an argument of decode().
"""
import torch
from torch import nn
from .math_ops import ste_round, local_snr
from .semantic import MapPredictor, QuantizerControl, TextCrossAttention


class SemanticEntropy(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        # Own new modules only. The backbone already registers the original codec.
        object.__setattr__(self, '_base', base)
        self.cfg = cfg
        self.student = MapPredictor(base.M, hidden=cfg.hidden)
        self.decoder_maps = MapPredictor(base.M, hidden=cfg.hidden)
        self.control = QuantizerControl(base.M, hidden=cfg.hidden)
        self.text_context = nn.ModuleList([TextCrossAttention(base.M * 2, hidden=cfg.hidden) for _ in range(4)])

    @property
    def base(self):
        return self._base

    @property
    def adaptive(self):
        return self.cfg.method in {'cadc_style', 'idea1', 'joint'}

    def effective_flags(self):
        from copy import deepcopy
        f = deepcopy(self.cfg.components)
        if self.cfg.method == 'cadc_style':
            for k in ('importance', 'difficulty', 'explanation', 'correction', 'text_entropy'):
                setattr(f, k, False)
        return f

    def prepare(self, latent, aux, text, mask):
        b = self.base
        y = b.g_a(latent, aux)
        z = b.h_a(y)  # Hyperprior sees unscaled y; always sent first.
        return y, z

    def _loop(self, zhat, text, text_mask, y=None, operation='forward', stream=None):
        b, f = self.base, self.effective_flags()
        base = b.h_s(zhat)
        batch, channels, height, width = base.shape
        if y is not None and base.shape != y.shape:
            raise ValueError('Hyperprior shape mismatch: pad input to a multiple of 256')
        if operation != 'forward' and batch != 1:
            raise ValueError('Real bitstream coding requires batch_size=1')
        masks = b.get_mask_four_parts(batch, channels, height, width, device=base.device)
        maps, scale = None, None
        if y is not None:
            maps = self.student(y, text, text_mask, self.cfg.quality, f.learned_token_weights) if self.cfg.method != 'cadc_style' else {}
            scale = self.control(y, base, maps, self.cfg.quality, f) if self.adaptive else torch.ones_like(y[:, :1])
            y = y / scale
        conditional = self.cfg.method in {'idea1', 'joint'} and f.text_entropy
        groups, likelihoods, mus, sigmas, raw_scales = [], [], [], [], []
        if operation in {'encode', 'decode'}:
            from compressai.ans import BufferedRansEncoder, RansDecoder
            cdf = b.gaussian_conditional.quantized_cdf.tolist()
            lengths = b.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets = b.gaussian_conditional.offset.reshape(-1).int().tolist()
            if not cdf:
                raise RuntimeError('CDF tables empty: call model.update() before encoding/decoding')
            symbols, indexes = [], []
            coder = BufferedRansEncoder() if operation == 'encode' else RansDecoder()
            if operation == 'decode':
                coder.set_stream(stream)
        for k, m in enumerate(masks):
            h = b.g_c(b.adapter_in[k](base))
            if conditional:
                h = self.text_context[k](h, text, text_mask)
            mu, sigma_raw = b.adapter_out[k](h).chunk(2, 1)
            # Original GC lower bound is 0.11. Use the identical positive effective scale.
            sigma = b.gaussian_conditional.lower_bound_scale(sigma_raw)
            if operation == 'forward':
                part, prob = b.forward_with_mask(y, sigma, mu, m)
                likelihoods.append(prob)
            elif operation == 'encode':
                part = b.compress_group_with_mask(b.gaussian_conditional, y, sigma, mu, m, symbols, indexes)
            else:
                part = b.decompress_group_with_mask(b.gaussian_conditional, sigma, mu, m, coder, cdf, lengths, offsets)
            # Keep original variance decoder input, including original raw pre-bound values.
            raw_scales.append(sigma_raw * m)
            mus.append(mu * m)
            sigmas.append(sigma * m)
            part = part + .5 * torch.tanh(b.LRP[k](torch.cat([part, base], 1)) * m)
            groups.append(part)
            base = base * (1 - m) + part
        qhat = sum(groups)
        out = {'mean': b.g_s(qhat), 'scale': b.scale(sum(raw_scales)),
               'res': b.aux(qhat), 'prompt': b.prompt(qhat),
               'qhat': qhat, 'snr': local_snr(sum(mus), sum(sigmas)),
               'maps': maps, 'm': scale}
        if operation == 'forward':
            out['y_likelihoods'] = sum(likelihoods)
        if operation == 'encode':
            coder.encode_with_indexes(symbols, indexes, cdf, lengths, offsets)
            out['y_string'] = coder.flush()
        return out

    def forward(self, latent, aux, text, mask):
        b = self.base
        y, z = self.prepare(latent, aux, text, mask)
        _, zprob = b.entropy_bottleneck(z)
        medians = b.entropy_bottleneck._get_medians()
        zhat = ste_round(z - medians) + medians
        out = self._loop(zhat, text, mask, y, 'forward')
        out['likelihoods'] = {'y': out.pop('y_likelihoods'), 'z': zprob}
        return out

    @torch.no_grad()
    def compress(self, latent, aux, text, mask):
        b = self.base
        y, z = self.prepare(latent, aux, text, mask)
        zs = b.entropy_bottleneck.compress(z)
        zshape = tuple(z.shape[-2:])
        zhat = b.entropy_bottleneck.decompress(zs, zshape)
        out = self._loop(zhat, text, mask, y, 'encode')
        return {'strings': [[out['y_string']], zs], 'shape': zshape}

    @torch.no_grad()
    def decompress(self, strings, shape, text, mask):
        # Deliberately no y, original image, teacher, m or encoder maps accepted.
        zhat = self.base.entropy_bottleneck.decompress(strings[1], shape)
        return self._loop(zhat, text, mask, operation='decode', stream=strings[0][0])

"""Inference and independent byte-stream decoding. No image is accepted by decode_file."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from PIL import Image
import torch
from .bitstream import pack, unpack, rate_report, MAX_PAYLOAD, MAX_HEADER
from .checkpoint import read_checkpoint, load_weights, EMA
from .data import image_tensor, jsonl_read
from .math_ops import pad_image
from .model import AIGICodec
from .text import TextEncoder
from .utils import sha256_file, canonical_hash, runtime_fingerprint, atomic_json, seed_all


class CodecSession:
    def __init__(self, cfg, checkpoint=None, device='cuda', use_ema=False):
        if cfg.method != 'ditic' and checkpoint is None:
            raise ValueError('New methods require a trained AIGI01 --checkpoint; random adapters are not a result')
        self.cfg, self.device = cfg, device
        seed_all(cfg.seed, cfg.deterministic)
        if cfg.stage != 'codec':
            raise ValueError('Map-training checkpoints/configs cannot encode images; use a codec-stage config')
        self.model = AIGICodec(cfg, device, training=bool(checkpoint))
        if checkpoint:
            state = read_checkpoint(checkpoint)
            for key in ('method', 'quality', 'components', 'dit_rank', 'vae_rank', 'finetune'):
                if cfg.to_dict()[key] != state['config'][key]:
                    raise ValueError('Inference checkpoint/config mismatch: ' + key)
            load_weights(self.model, state['model'])
            if use_ema:
                if not state['ema']:
                    raise ValueError('Checkpoint has no EMA shadow')
                shadow = EMA(self.model)
                shadow.shadow = state['ema']
                shadow.apply(self.model)
        self.model.eval().requires_grad_(False)
        self.model.update()
        self.text = None if cfg.method in {'ditic', 'cadc_style'} else TextEncoder(cfg.sana_path, cfg.offline, cfg.text_length)
        self.text_id = self.text.fingerprint if self.text else '0' * 64
        source_sha = sha256_file(checkpoint or cfg.weights)
        source_code = canonical_hash({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        self.model_id = canonical_hash(dict(weights=source_sha, ema=use_ema, method=cfg.method,
                                            quality=cfg.quality, components=cfg.to_dict()['components'],
                                            beta=cfg.beta, text_length=cfg.text_length, code_sha256=source_code, format='aigi01-v1'))
        self.runtime = runtime_fingerprint()

    def get_text(self, prompt):
        if self.text:
            return self.text.paired([prompt], self.device)
        z = torch.zeros(1, 77, 2304, device=self.device)
        mask = torch.ones(1, 77, device=self.device, dtype=torch.bool)
        return z, mask, z, mask

    def sync(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()

    @torch.no_grad()
    def encode_file(self, image_path, output, prompt='', external_prompt=False):
        x = image_tensor(image_path).unsqueeze(0).to(self.device)
        x, (h, w) = pad_image(x)
        if h > 8192 or w > 8192:
            raise ValueError('AIGI01 v1 supports dimensions up to 8192')
        # Baseline consumes no text. The common frame still has an empty zlib prompt field.
        if self.text is None:
            prompt = ''
        text, mask, _, _ = self.get_text(prompt)
        self.sync()
        begin = time.perf_counter()
        code = self.model.compress(x, text, mask)
        self.sync()
        encode_ms = (time.perf_counter() - begin) * 1000
        metadata = dict(method=self.cfg.method, quality=self.cfg.quality, height=h, width=w,
                        padded_height=x.shape[-2], padded_width=x.shape[-1], shape=list(code['shape']),
                        model_sha256=self.model_id, text_sha256=self.text_id, runtime=self.runtime)
        data = pack(metadata, prompt, code['strings'], external_prompt)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        return dict(rate_report(data), encode_ms=encode_ms, bitstream=str(output.resolve()))

    @torch.no_grad()
    def decode_file(self, input_path, output, external_prompt=None):
        path = Path(input_path)
        if path.stat().st_size > MAX_PAYLOAD + MAX_HEADER + 44:
            raise ValueError('Input bitstream too large')
        packet = unpack(path.read_bytes(), external_prompt)
        meta = packet.metadata
        for key, value in (('model_sha256', self.model_id), ('text_sha256', self.text_id),
                           ('method', self.cfg.method), ('quality', self.cfg.quality), ('runtime', self.runtime)):
            if meta[key] != value:
                raise ValueError('Decoder contract mismatch: ' + key)
        text, mask, null, nmask = self.get_text(packet.prompt)
        self.sync()
        begin = time.perf_counter()
        image, extra = self.model.decompress(packet.strings, meta['shape'], text, mask, null, nmask)
        self.sync()
        decode_ms = (time.perf_counter() - begin) * 1000
        image = image[0, :, :meta['height'], :meta['width']]
        pixels = ((image.clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)
        return {'decode_ms': decode_ms, 'nfe': extra['nfe'], 'reconstruction': str(output.resolve())}


def infer_manifest(session, manifest, output, limit=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, row in enumerate(jsonl_read(manifest)):
        if limit and i >= limit:
            break
        path = Path(row['image'])
        if not path.is_absolute():
            path = Path(manifest).resolve().parent / path
        if sha256_file(path) != row['sha256']:
            raise ValueError('Reference image changed after manifest creation')
        ident = row['id']
        if not isinstance(ident, str) or not ident.isalnum():
            raise ValueError('Manifest IDs must be nonempty alphanumeric strings')
        codepath = output / 'bitstreams' / f'{ident}.aigi'
        reconpath = output / 'reconstructions' / f'{ident}.png'
        enc = session.encode_file(path, codepath, row.get('prompt', ''))
        dec = session.decode_file(codepath, reconpath)
        result = dict(enc, **dec, id=ident, original=str(path.resolve()), image_sha256=row['sha256'],
                      dataset=row.get('dataset'), method=session.cfg.method, quality=session.cfg.quality,
                      reconstruction_sha256=sha256_file(reconpath), bitstream_sha256=sha256_file(codepath))
        results.append(result)
        print(json.dumps(result), flush=True)
    if not results:
        raise ValueError('Empty inference manifest')
    atomic_json({'model_sha256': session.model_id, 'text_sha256': session.text_id, 'runtime': session.runtime,
                 'method': session.cfg.method, 'quality': session.cfg.quality, 'samples': results,
                 'bpp_total': sum(x['total_bits'] for x in results) / sum(x['pixels'] for x in results),
                 'bpp_mean': sum(x['bpp_total'] for x in results) / len(results),
                 'experiment': session.cfg.experiment or session.cfg.method,
                 'note': 'Timing excludes text encoding/model loading and includes CPU-reference entropy transfers.'}, output / 'inference.json')
    return str(output / 'inference.json')

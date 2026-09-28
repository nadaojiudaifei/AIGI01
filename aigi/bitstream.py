"""Versioned, bounded, checksummed binary container. Never uses pickle for input streams."""
from dataclasses import dataclass
import hashlib
import json
import struct
import zlib

MAGIC = b'AIGI01\x00\x01'
MAX_HEADER = 64 * 1024
MAX_PROMPT = 64 * 1024
MAX_PAYLOAD = 256 * 1024 * 1024
MAX_PIXELS = 8192 * 8192


@dataclass
class Packet:
    metadata: dict
    prompt: str
    strings: list


def _valid(meta):
    required = {'method', 'quality', 'height', 'width', 'padded_height', 'padded_width',
                'shape', 'model_sha256', 'text_sha256', 'prompt_mode', 'prompt_sha256',
                'lengths', 'prompt_bytes', 'runtime'}
    if not isinstance(meta, dict) or set(meta) != required:
        raise ValueError('Unknown/missing bitstream fields')
    if meta['method'] not in {'ditic', 'cadc_style', 'idea1', 'idea2', 'joint'}:
        raise ValueError('Unsupported codec method')
    if type(meta['quality']) is not int or meta['quality'] not in (1, 2, 3, 4):
        raise ValueError('Invalid quality')
    for key in ('height', 'width', 'padded_height', 'padded_width'):
        if type(meta[key]) is not int or not 1 <= meta[key] <= 8192:
            raise ValueError('Invalid image dimension')
    h, w, ph, pw = (meta[k] for k in ('height', 'width', 'padded_height', 'padded_width'))
    if h * w > MAX_PIXELS or ph != (h + 255) // 256 * 256 or pw != (w + 255) // 256 * 256:
        raise ValueError('Invalid padding')
    if meta['shape'] != [ph // 256, pw // 256]:
        raise ValueError('Invalid hyperprior shape')
    for k in ('model_sha256', 'text_sha256', 'prompt_sha256'):
        v = meta[k]
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('Invalid digest')
    if meta['prompt_mode'] not in ('inline', 'external'):
        raise ValueError('Invalid prompt mode')
    if not isinstance(meta['runtime'], str) or len(meta['runtime']) > 256:
        raise ValueError('Invalid runtime fingerprint')
    lens = meta['lengths']
    if not isinstance(lens, list) or len(lens) != 2 or any(type(n) is not int or not 0 < n <= MAX_PAYLOAD for n in lens):
        raise ValueError('Exactly one main and one hyper stream required')
    if type(meta['prompt_bytes']) is not int or not 0 <= meta['prompt_bytes'] <= MAX_PROMPT + 128:
        raise ValueError('Invalid prompt length')
    if sum(lens) + meta['prompt_bytes'] > MAX_PAYLOAD:
        raise ValueError('Payload too large')


def pack(metadata, prompt, strings, external_prompt=False):
    if len(strings) != 2 or any(len(group) != 1 for group in strings):
        raise ValueError('One image per packet')
    raw = prompt.encode('utf-8')
    if len(raw) > MAX_PROMPT:
        raise ValueError('Prompt exceeds 64 KiB')
    compressed = b'' if external_prompt else zlib.compress(raw, 9)
    meta = dict(metadata, prompt_mode='external' if external_prompt else 'inline',
                prompt_sha256=hashlib.sha256(raw).hexdigest(), prompt_bytes=len(compressed),
                lengths=[len(strings[0][0]), len(strings[1][0])])
    _valid(meta)
    header = json.dumps(meta, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
    if len(header) > MAX_HEADER:
        raise ValueError('Header too long')
    body = MAGIC + struct.pack('>I', len(header)) + header + compressed + strings[0][0] + strings[1][0]
    return body + hashlib.sha256(body).digest()


def _envelope(data):
    if not isinstance(data, bytes) or len(data) < 44 or len(data) > MAX_PAYLOAD + MAX_HEADER + 44:
        raise ValueError('Invalid packet size')
    if data[:8] != MAGIC:
        raise ValueError('Not an AIGI01 v1 stream')
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('Corrupt/truncated stream (SHA256 mismatch)')
    n = struct.unpack('>I', data[8:12])[0]
    if n > MAX_HEADER or 12 + n > len(data) - 32:
        raise ValueError('Invalid header size')
    meta = json.loads(data[12:12+n])
    _valid(meta)
    offset = 12 + n
    lens = meta['lengths']
    pbytes = meta['prompt_bytes']
    if offset + pbytes + sum(lens) != len(data) - 32:
        raise ValueError('Trailing/missing payload bytes')
    return meta, offset, lens, pbytes


def unpack(data, external_prompt=None):
    meta, offset, lens, pbytes = _envelope(data)
    if meta['prompt_mode'] == 'external':
        if pbytes or external_prompt is None:
            raise ValueError('This stream requires the exact external prompt')
        raw = external_prompt.encode('utf-8')
        if len(raw) > MAX_PROMPT:
            raise ValueError('External prompt exceeds 64 KiB')
    else:
        dec = zlib.decompressobj()
        raw = dec.decompress(data[offset:offset+pbytes], MAX_PROMPT + 1)
        if len(raw) > MAX_PROMPT or dec.unconsumed_tail or not dec.eof or dec.unused_data:
            raise ValueError('Invalid/oversized compressed prompt')
    if hashlib.sha256(raw).hexdigest() != meta['prompt_sha256']:
        raise ValueError('Prompt hash mismatch')
    offset += pbytes
    strings = [[data[offset:offset+lens[0]]], [data[offset+lens[0]:offset+sum(lens)]]]
    return Packet(meta, raw.decode('utf-8'), strings)


def rate_report(data):
    meta, _, _, _ = _envelope(data)
    if meta['prompt_mode'] == 'inline':
        unpack(data)  # Also validate bounded zlib and prompt digest.
    pixels = meta['height'] * meta['width']
    payload = sum(meta['lengths'])
    return {'total_bits': len(data) * 8, 'pixels': pixels,
            'bpp_total': len(data) * 8 / pixels,
            'bpp_entropy_payload': payload * 8 / pixels,
            'bpp_prompt': meta['prompt_bytes'] * 8 / pixels,
            'bpp_framing': (len(data) - payload - meta['prompt_bytes']) * 8 / pixels}

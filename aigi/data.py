"""Local manifest I/O; this module intentionally never imports Hugging Face datasets."""
from pathlib import Path
import hashlib
import json
import os
import unicodedata
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from .utils import sha256_file, jsonl_read, canonical_hash

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def open_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def image_tensor(path, crop=0):
    im = open_rgb(path)
    a = np.array(im, dtype=np.uint8)
    if crop:
        h, w = a.shape[:2]
        a = np.pad(a, ((0, max(0, crop-h)), (0, max(0, crop-w)), (0, 0)), mode='edge')
        h, w = a.shape[:2]
        top, left = (h-crop)//2, (w-crop)//2
        a = a[top:top+crop, left:left+crop]
    return torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 127.5 - 1.


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def canonical_prompt(prompt):
    return ' '.join(unicodedata.normalize('NFC', prompt).casefold().split())


def group_key(row):
    prompt = canonical_prompt(row.get('prompt', ''))
    if prompt and row.get('prompt_origin') in {'original', 'paired_caption'}:
        return 'prompt:' + hashlib.sha256(prompt.encode()).hexdigest()
    return 'pixels:' + row['pixel_sha256']


def record_image(path, root, dataset, prompt='', origin='missing', kind='natural', source_split='train'):
    path = Path(path).resolve()
    im = open_rgb(path)
    pixel = pixel_hash(im)
    return dict(id=canonical_hash([dataset, str(path.relative_to(root)), pixel])[:24],
                image=str(path), sha256=sha256_file(path), pixel_sha256=pixel,
                prompt=prompt, prompt_origin=origin, dataset=dataset, kind=kind,
                width=im.width, height=im.height, source_split=source_split)


def build_local_manifest(root, output, dataset='local', kind='natural', source_split='train', expected=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    metadata = {}
    # Official DiffusionDB ZIP files contain part-NNNNNN.json with p or prompt.
    for p in sorted(root.rglob('part-*.json')):
        raw = json.loads(p.read_text())
        for name, val in raw.items():
            prompt = val.get('p', val.get('prompt', ''))
            metadata[name] = (prompt, 'original')
    for p in sorted(root.rglob('metadata.jsonl')):
        for row in jsonl_read(p):
            name = row.get('file_name', row.get('image', ''))
            metadata[str((p.parent / name).resolve())] = (row.get('prompt', row.get('text', '')), 'original' if kind == 'aigi' else 'caption')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if expected is not None and len(paths) != expected:
        raise ValueError(f'Expected {expected} images but found {len(paths)} in {root}')
    if not paths:
        raise ValueError('No images found')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('w', encoding='utf-8') as f:
        for p in paths:
            prompt, origin = metadata.get(str(p.resolve()), metadata.get(p.name, ('', 'missing')))
            txt = p.with_suffix('.txt')
            if not prompt and txt.is_file():
                prompt = txt.read_text(encoding='utf-8').strip()
                origin = 'original' if kind == 'aigi' else 'caption'
            row = record_image(p, root, dataset, prompt, origin, kind, source_split)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def split_manifests(inputs, output, seed=903):
    """Connected-component 98/1/1 split by identical pixels OR original prompts.

    Union-find closes transitive duplicates: A/B share pixels, B/C share a prompt.
    Any official evaluation member reserves the ENTIRE connected component.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in inputs:
        for row in jsonl_read(path):
            image = Path(row['image'])
            if not image.is_absolute():
                row = dict(row, image=str((Path(path).resolve().parent / image).resolve()))
            rows.append(row)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a,b)] = min(a,b)
    seen_pixels, seen_groups = {}, {}
    for i, row in enumerate(rows):
        for table, key in ((seen_pixels, row['pixel_sha256']), (seen_groups, group_key(row))):
            if key in table:
                union(i, table[key])
            else:
                table[key] = i
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(find(i), []).append(row)
    handles = {k: (out / f'{k}.jsonl').open('w', encoding='utf-8') for k in ('train','val','test')}
    counts = dict.fromkeys(handles, 0)
    unique_pixels = set()
    try:
        for members in grouped.values():
            group = min(group_key(r) for r in members)
            sources = {r.get('source_split','train') for r in members}
            bucket = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8],16) % 10000
            split = ('test' if 'test' in sources else 'val' if sources & {'val','validation'}
                     else 'train' if bucket < 9800 else 'val' if bucket < 9900 else 'test')
            for row in members:
                if row['pixel_sha256'] in unique_pixels:
                    continue
                unique_pixels.add(row['pixel_sha256'])
                handles[split].write(json.dumps(dict(row,group=group,split=split),ensure_ascii=False)+'\n')
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (out / 'split_report.json').write_text(json.dumps(dict(seed=seed,counts=counts,components=len(grouped)),indent=2))
    return counts


def assert_disjoint(train_path, val_path):
    train = list(jsonl_read(train_path))
    val = list(jsonl_read(val_path))
    if not train or not val:
        raise ValueError('Train and validation manifests must both be nonempty')
    for key in ('pixel_sha256',):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError('Train/validation pixel leakage detected')
    if {group_key(r) for r in train} & {group_key(r) for r in val}:
        raise ValueError('Train/validation prompt-group leakage detected')


def cache_key(row, crop, quality):
    return canonical_hash({'image': row['sha256'], 'prompt': row.get('prompt', ''),
                           'crop': crop, 'quality': quality, 'version': 'teacher-v1'})


class ManifestDataset(Dataset):
    def __init__(self, path, crop=0, cache=None, quality=3, require_cache=False,
                 augmentation='center', views_per_image=1, seed=903):
        self.path = Path(path).resolve()
        self.rows = list(jsonl_read(self.path))
        if not self.rows:
            raise ValueError('Empty manifest: ' + str(path))
        self.crop, self.cache, self.quality = crop, Path(cache) if cache else None, quality
        self.require_cache = require_cache
        self.augmentation, self.views, self.seed = augmentation, views_per_image, seed
        if augmentation not in {'center','cached_random'} or views_per_image < 1:
            raise ValueError('Invalid augmentation/views')
        if augmentation == 'center' and views_per_image != 1:
            raise ValueError('Center policy has exactly one view')
        if augmentation != 'center' and not crop:
            raise ValueError('Random augmentation requires crop')
        for row in self.rows:
            p = Path(row['image'])
            if not p.is_absolute():
                p = self.path.parent / p
            row['image'] = str(p.resolve())
        self._verified = set()
        self._teacher_fingerprint = None
        if self.cache:
            contract_path = self.cache / 'cache_contract.json'
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text())
                self._teacher_fingerprint = (contract['teacher_sha256'], contract['text_sha256'])
            elif require_cache:
                raise FileNotFoundError('Teacher cache contract missing; regenerate cache with teacher command')

    def __len__(self):
        return len(self.rows) * self.views

    def row_at(self, i):
        return self.rows[i // self.views]

    def spec_at(self, i):
        from .augmentation import view_spec
        return view_spec(self.row_at(i), self.crop, i % self.views, self.seed, self.augmentation)

    def key_at(self, i):
        key = cache_key(self.row_at(i), self.crop, self.quality)
        return key if self.augmentation == 'center' else canonical_hash({'base': key, 'view': self.spec_at(i)})

    def __getitem__(self, i):
        row = self.row_at(i)
        path = row['image']
        if path not in self._verified:
            if sha256_file(path) != row['sha256']:
                raise ValueError('Image changed after manifest export: ' + path)
            self._verified.add(path)
        from .augmentation import apply_view
        image = (apply_view(open_rgb(path), self.spec_at(i)) if self.augmentation != 'center'
                 else image_tensor(path, self.crop))
        item = {'image': image, 'prompt': row.get('prompt', ''),
                'id': row['id'], 'row': row}
        if self.cache:
            key = self.key_at(i)
            cachepath = self.cache / (key + '.npz')
            if cachepath.is_file():
                with np.load(cachepath, allow_pickle=False) as record:
                    meta = json.loads(str(record['metadata']))
                    if meta.get('cache_key') != key:
                        raise ValueError('Teacher cache does not match image/prompt/crop/quality')
                    fingerprint = (meta.get('teacher_sha256'), meta.get('text_sha256'))
                    if self._teacher_fingerprint is not None and self._teacher_fingerprint != fingerprint:
                        raise ValueError('Mixed teacher/text assets in one dataset cache')
                    self._teacher_fingerprint = fingerprint
                    for name in ('attention', 's', 'c', 'e', 'text', 'mask', 'null_text', 'null_mask'):
                        item[name] = torch.from_numpy(record[name].copy())
            elif self.require_cache:
                raise FileNotFoundError(f'Missing teacher cache {cachepath}; run teacher command')
        return item


def collate(items):
    keys = set(items[0])
    if any(set(i) != keys for i in items):
        raise ValueError('Mixed cached/uncached samples in a batch')
    return {k: torch.stack([i[k] for i in items]) if torch.is_tensor(items[0][k]) else [i[k] for i in items] for k in keys}

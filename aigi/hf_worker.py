"""Standalone isolated worker. Do not import aigi or add the upstream root to sys.path."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    args = p.parse_args()
    r = json.loads(Path(args.request).read_text())
    # The script directory is aigi/, not the root containing original datasets/.
    from datasets import load_dataset, load_from_disk, Image as HFImage
    from huggingface_hub import HfApi
    from PIL import Image, ImageOps
    source = r.get('local_source')
    revision = r.get('revision')
    if source:
        source = Path(source).resolve()
        if (source / 'dataset_dict.json').exists() or (source / 'state.json').exists():
            ds = load_from_disk(str(source))
            if hasattr(ds, 'keys'):
                ds = ds[r['split']]
        elif list(source.rglob('*.parquet')):
            paths = sorted(str(p) for p in source.rglob('*.parquet') if r['split'] in str(p.relative_to(source)))
            if not paths:
                raise ValueError('No split-matching parquet files; export with the correct --split')
            ds = load_dataset('parquet', data_files={r['split']: paths}, split=r['split'], streaming=True)
        else:
            ds = load_dataset('imagefolder', data_dir=str(source), split=r['split'], streaming=True)
    else:
        if not revision:
            revision = HfApi().dataset_info(r['repo']).sha
        ds = load_dataset(r['repo'], name=r.get('config'), split=r['split'], revision=revision,
                          streaming=True)  # No trust_remote_code; script-only repos use explicit adapters.
    image_key, prompt_key = r['image'], r.get('prompt')
    if ds.features is None:
        ds = ds.resolve_features() if hasattr(ds, 'resolve_features') else ds._resolve_features()
    if image_key not in ds.features:
        raise ValueError(f'Missing image column {image_key}; available {list(ds.features)}')
    if isinstance(ds.features[image_key], HFImage):
        ds = ds.cast_column(image_key, HFImage(decode=False))
    out = Path(r['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    images = out / 'images'
    images.mkdir(exist_ok=True)
    count = 0
    questions = []
    with (out / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for i, row in enumerate(ds):
            if r.get('limit') and count >= r['limit']:
                break
            # Filter only when explicit NSFW metadata exists; no hidden detector.
            if float(row.get('image_nsfw', 0) or 0) > r.get('max_nsfw', .1) or float(row.get('prompt_nsfw', 0) or 0) > r.get('max_nsfw', .1):
                continue
            value = row[image_key]
            if isinstance(value, dict):
                payload = value.get('bytes')
                im = Image.open(io.BytesIO(payload) if payload is not None else value['path'])
            elif isinstance(value, (str, Path)):
                im = Image.open(value)
            else:
                im = value
            im = ImageOps.exif_transpose(im).convert('RGB')
            target = images / f'{i:09d}.png'
            im.save(target)
            filehash = hashlib.sha256(target.read_bytes()).hexdigest()
            pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            prompt = str(row.get(prompt_key, '') or '') if prompt_key else ''
            generator = str(row.get('generator', ''))
            kind = r['kind']
            if kind == 'mixed':
                kind = 'natural' if row.get('label') == 0 or generator == 'real' else 'aigi'
            origin = ('original' if kind == 'aigi' else 'paired_caption') if prompt else 'missing'
            record = dict(id=hashlib.sha256(f'{r["repo"]}:{i}:{pixel}'.encode()).hexdigest()[:24],
                          image=str(target), sha256=filehash, pixel_sha256=pixel, prompt=prompt,
                          prompt_origin=origin, generator=generator, dataset=r['repo'], kind=kind,
                          width=im.width, height=im.height, source_split=r['split'], revision=revision)
            if r.get('ocrbench'):
                annotation = {k:v for k,v in row.items() if k != image_key and v is not None}
                if annotation.get('eval') in ('None','none','nan',''): annotation.pop('eval',None)
                annotation['image_path'] = str(target.relative_to(out))
                questions.append(annotation)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    if r.get('ocrbench'):
        (out/'OCRBench_v2.json').write_text(json.dumps(questions,ensure_ascii=False),encoding='utf-8')
    (out / 'export.json').write_text(json.dumps(dict(request=r, resolved_revision=revision, exported=count), indent=2))
    if not count:
        raise ValueError('No examples exported')
    print(json.dumps(dict(exported=count, manifest=str(out / 'manifest.jsonl'))))


if __name__ == '__main__':
    main()

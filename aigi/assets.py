"""Explicit downloads only. Training/inference never silently fetch data or model assets."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from .registry import DATASETS, MODELS
from .utils import atomic_json, sha256_file


def safe_unzip(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe ZIP member: ' + member.filename)
            if member.file_size > 256 * 1024 * 1024:
                raise ValueError('Unexpected oversized image/metadata ZIP member')
        z.extractall(destination)


def export_hf(name, output, split=None, limit=0, local_source=None, revision=None, config=None):
    if name not in DATASETS or DATASETS[name]['loader'] != 'hf':
        raise ValueError('This catalog entry requires its explicit archive adapter')
    r = dict(DATASETS[name], output=str(Path(output).resolve()), limit=limit,
             local_source=str(Path(local_source).resolve()) if local_source else None, revision=revision, config=config)
    if split:
        r['split'] = split
    worker = Path(__file__).with_name('hf_worker.py').resolve()
    with tempfile.TemporaryDirectory(prefix='aigi_hf_') as td:
        request = Path(td) / 'request.json'
        request.write_text(json.dumps(r))
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, '-I', str(worker), '--request', str(request)], cwd=td, env=env, check=True)


def diffusiondb(name, output, first=None, last=None, local_source=None, revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    from .data import build_local_manifest
    spec = DATASETS[name]
    first, last = first or spec['first'], last or spec['last']
    max_part = 14000 if spec['large'] else 2000
    if not 1 <= first <= last <= max_part:
        raise ValueError('Invalid DiffusionDB part range')
    output = Path(output).resolve()
    archive_dir, image_dir = output / 'archives', output / 'unpacked'
    archive_dir.mkdir(parents=True, exist_ok=True)
    if not local_source:
        revision = revision or HfApi().dataset_info(spec['repo']).sha
    for n in range(first, last + 1):
        folder = ('diffusiondb-large-part-1' if n <= 10000 else 'diffusiondb-large-part-2') if spec['large'] else 'images'
        filename = f'{folder}/part-{n:06d}.zip'
        if local_source:
            archive = Path(local_source) / filename
            if not archive.is_file():
                archive = Path(local_source) / Path(filename).name
            if not archive.is_file():
                raise FileNotFoundError(archive)
        else:
            archive = hf_hub_download(spec['repo'], filename, repo_type='dataset', revision=revision, local_dir=str(archive_dir))
        safe_unzip(archive, image_dir / f'part-{n:06d}')
    count = build_local_manifest(image_dir, output / 'manifest.jsonl', name, 'aigi')
    atomic_json(dict(repo=spec['repo'], revision=revision, first=first, last=last, exported=count,
                     note='Official part ZIP has no NSFW scores; apply metadata filtering before research use.'), output / 'export.json')
    return count


def mlic_download(output, local_source=None, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    from .data import build_local_manifest
    output = Path(output).resolve()
    spec = DATASETS['mlic100k']
    if local_source:
        source = Path(local_source).resolve()
    else:
        revision = revision or HfApi().dataset_info(spec['repo']).sha
        source = Path(snapshot_download(spec['repo'], repo_type='dataset', revision=revision,
                                       allow_patterns=spec['patterns'], local_dir=str(output / 'archives')))
    volumes = sorted(source.glob('train512x512.7z.*'))
    if len(volumes) != 37 or volumes[0].suffix != '.001':
        raise ValueError('MLIC requires all 37 volumes, starting at .001')
    if not shutil.which('7z'):
        raise RuntimeError('7z not installed; install p7zip in your remote environment')
    listing = subprocess.check_output(['7z', 'l', '-slt', str(volumes[0])], text=True)
    content = listing.split('----------', 1)[-1]
    for line in content.splitlines():
        if line.startswith(('Symbolic Link = ', 'Hard Link = ')):
            raise ValueError('Archive links are not accepted')
        if line.startswith('Path = '):
            name = line[7:]
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe 7z member')
    destination = output / 'unpacked'
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(volumes[0]), '-o' + str(destination), '-y'], check=True)
    return build_local_manifest(destination, output / 'manifest.jsonl', 'mlic100k', 'natural')


def download_models(output, qualities=(3,), include_caption=True, plan=False):
    from huggingface_hub import HfApi, snapshot_download
    out = Path(output).resolve()
    api, records = HfApi(), {}
    for name, repo in MODELS.items():
        if name == 'caption' and not include_caption:
            continue
        info = api.model_info(repo, files_metadata=True)
        files = [s.rfilename for s in info.siblings]
        if name == 'ditic':
            desired = [f'q{q}_merge_ema.pt' for q in qualities] + ['elic_official.pth']
            selected = []
            for item in desired:
                matches = [f for f in files if Path(f).name == item]
                if len(matches) != 1:
                    raise ValueError(f'{repo}: expected one {item}, found {matches}. Inspect model files before downloading.')
                selected += matches
        else:
            selected = [f for f in files if not f.endswith(('.md', '.png', '.jpg', '.gitattributes'))]
        sizes = {s.rfilename: int(s.size or 0) for s in info.siblings}
        records[name] = dict(repo=repo, revision=info.sha, files=selected, bytes=sum(sizes[f] for f in selected))
        if not plan:
            location = snapshot_download(repo, revision=info.sha, allow_patterns=selected, local_dir=str(out / name))
            records[name]['local_path'] = location
    if plan:
        print(json.dumps(records, indent=2))
    else:
        atomic_json(records, out / 'assets.lock.json')
    return records


def download_url(url, destination, expected_sha256):
    if not url.startswith('https://') or len(expected_sha256) != 64:
        raise ValueError('HTTPS and explicit SHA256 required for external/GitHub release model assets')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.part')
    with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as f:
        shutil.copyfileobj(response, f)
    if sha256_file(temp) != expected_sha256:
        temp.unlink()
        raise ValueError('Downloaded file SHA256 mismatch')
    os.replace(temp, destination)


def configure(cfg, assets):
    root = Path(assets).resolve()
    def unique(name):
        matches = list((root / 'ditic').rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one {name}, found {matches}')
        return str(matches[0])
    cfg.sana_path = str(root / 'sana')
    cfg.weights = unique(f'q{cfg.quality}_merge_ema.pt')
    cfg.elic_path = unique('elic_official.pth')
    return cfg

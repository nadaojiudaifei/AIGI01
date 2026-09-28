import hashlib
import json
from pathlib import Path
import os
import random
import tempfile
import numpy as np
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=p.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, p)


def seed_all(seed, deterministic=True):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(True)


def runtime_fingerprint():
    from importlib.metadata import version
    return ';'.join(f'{n}={version(n)}' for n in ('torch', 'compressai', 'diffusers', 'transformers')) + ';cpu-fp32-v1'


def jsonl_read(path):
    p = Path(path).resolve()
    with p.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{number}: invalid JSON') from e
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected object')
            yield row


def preload_hf_datasets():
    """Load HF datasets before original models/ imports can shadow it.

    This application uses its own manifest loader, never upstream datasets/.
    The isolated export worker handles the same collision in its subprocess.
    """
    import importlib
    import sys
    existing = sys.modules.get('datasets')
    if existing is not None:
        if not hasattr(existing, 'load_dataset'):
            raise RuntimeError('A non-Hugging-Face datasets module is already imported; start a fresh aigi process')
        return existing
    original = list(sys.path)
    try:
        sys.path[:] = [p for p in original if not (Path(p or '.').resolve() / 'models/DiT_IC.py').is_file()]
        module = importlib.import_module('datasets')
        if not hasattr(module, 'load_dataset'):
            raise RuntimeError('Could not resolve Hugging Face datasets; inspect Python path')
        return module
    finally:
        sys.path[:] = original

"""Deterministic finite-view augmentation, shared by teacher and all baselines.

Every view is identified by image hash, crop, view index and seed. Workers and
resume position cannot change it. The teacher observes the actual transformed
image; maps are never geometrically transformed under an unchanged text teacher.
"""
import hashlib
import random
import numpy as np
import torch


def view_spec(row, crop, view=0, seed=903, policy='center'):
    if policy not in {'center', 'cached_random'}:
        raise ValueError('Unknown augmentation policy')
    if crop <= 0 or view < 0:
        raise ValueError('crop must be positive and view nonnegative')
    h, w = max(row['height'], crop), max(row['width'], crop)
    if policy == 'center':
        if view:
            raise ValueError('Center policy has exactly one view')
        return {'top': (h-crop)//2, 'left': (w-crop)//2, 'flip_h': False,
                'flip_v': False, 'crop': crop, 'policy': policy, 'version': 1}
    key = f"{row['sha256']}:{crop}:{view}:{seed}".encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest(), 16))
    return {'top': rng.randrange(h-crop+1), 'left': rng.randrange(w-crop+1),
            'flip_h': rng.random() < .5, 'flip_v': rng.random() < .5,
            'crop': crop, 'policy': policy, 'version': 1, 'view': view, 'seed': seed}


def apply_view(image, spec):
    a = np.array(image, dtype=np.uint8)
    crop, top, left = spec['crop'], spec['top'], spec['left']
    h, w = a.shape[:2]
    a = np.pad(a, ((0, max(0,crop-h)), (0, max(0,crop-w)), (0,0)), mode='edge')
    if not 0 <= top <= a.shape[0]-crop or not 0 <= left <= a.shape[1]-crop:
        raise ValueError('View coordinates outside image')
    a = a[top:top+crop, left:left+crop]
    if spec['flip_h']: a = a[:, ::-1]
    if spec['flip_v']: a = a[::-1]
    return torch.from_numpy(a.copy()).permute(2,0,1).float()/127.5-1

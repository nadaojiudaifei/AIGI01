"""Named full-reference, no-reference and original patch-FID/KID metrics."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import socket
import numpy as np
import torch
from .data import image_tensor
from .utils import atomic_json, sha256_file
from .backbone import require_upstream


@contextmanager
def network_guard(offline):
    # Prevent hidden weight downloads during offline evaluation. Warm caches explicitly.
    if not offline:
        yield
        return
    original = socket.socket.connect
    def deny(*args, **kwargs):
        raise RuntimeError('Offline metric weight missing: run warmup while online first')
    socket.socket.connect = deny
    try:
        yield
    finally:
        socket.socket.connect = original


class Metrics:
    def __init__(self, names, device='cuda', offline=True):
        self.names, self.device, self.metrics = set(names), device, {}
        known = {'psnr', 'ms_ssim', 'lpips', 'dists', 'niqe', 'clipiqa', 'musiq', 'fid', 'kid'}
        if self.names - known:
            raise ValueError('Unknown metrics: ' + str(self.names - known))
        with network_guard(offline):
            for name in self.names - {'psnr', 'fid', 'kid', 'lpips'}:
                from .utils import preload_hf_datasets
                preload_hf_datasets()
                import pyiqa
                self.metrics[name] = pyiqa.create_metric(name, device=device).eval()
            if 'lpips' in self.names:
                import lpips
                self.metrics['lpips'] = lpips.LPIPS(net='alex').to(device).eval()
            self.fid = self.kid = None
            if 'fid' in self.names:
                from torchmetrics.image.fid import FrechetInceptionDistance
                self.fid = FrechetInceptionDistance().to(device)
            if 'kid' in self.names:
                from torchmetrics.image.kid import KernelInceptionDistance
                self.kid = KernelInceptionDistance(subset_size=1000, subsets=100).to(device)

    @torch.no_grad()
    def pair(self, original, recon):
        if original.shape != recon.shape:
            raise ValueError('Metric image sizes differ; do not silently resize')
        output = {}
        if 'psnr' in self.names:
            mse = float((original.double() - recon.double()).square().mean())
            output['psnr'] = -10 * math.log10(mse) if mse else None
            if not mse:
                output['psnr_status'] = 'infinite (identical images)'
        for name, metric in self.metrics.items():
            if name in {'niqe', 'clipiqa', 'musiq'}:
                value = metric(recon)
            elif name == 'lpips':
                value = metric(recon * 2 - 1, original * 2 - 1)
            else:
                value = metric(recon, original)
            scalar = float(value.mean())
            if not math.isfinite(scalar):
                raise FloatingPointError('Non-finite ' + name)
            output[name] = scalar
        return output


def evaluate(inference_json, output, names, device='cuda', upstream_root='.', offline=True):
    protocol = json.loads(Path(inference_json).read_text())
    rows = protocol['samples']
    if not rows:
        raise ValueError('No inferred samples')
    metrics = Metrics(names, device, offline)
    if metrics.fid is not None or metrics.kid is not None:
        require_upstream(upstream_root)
        from eval._update_patch_fid import update_patch_fid
    per_image = []
    with torch.no_grad(), network_guard(offline):
        for row in rows:
            if sha256_file(row['original']) != row['image_sha256']:
                raise ValueError('Original image changed since inference')
            original = ((image_tensor(row['original']) + 1) / 2).unsqueeze(0).to(device)
            recon = ((image_tensor(row['reconstruction']) + 1) / 2).unsqueeze(0).to(device)
            scores = metrics.pair(original, recon)
            if len(rows) >= 50 and (metrics.fid is not None or metrics.kid is not None):
                if min(original.shape[-2:]) < 256:
                    raise ValueError('Original patch-FID requires both dimensions >=256; do not resize silently')
                update_patch_fid(original, recon, fid_metric=metrics.fid, kid_metric=metrics.kid)
            per_image.append(dict(row, **scores))
    averages = {}
    for name in set(names) - {'fid', 'kid'}:
        vals = [row[name] for row in per_image if isinstance(row.get(name), (float, int))]
        averages[name] = float(np.mean(vals)) if len(vals) == len(per_image) else None
    if metrics.fid is not None:
        averages['fid'] = float(metrics.fid.compute()) if len(rows) >= 50 else None
    if metrics.kid is not None:
        n = sum(x.shape[0] for x in metrics.kid.real_features)
        if len(rows) >= 50 and n >= 1000:
            mean, std = metrics.kid.compute()
            averages.update(kid_mean=float(mean), kid_std=float(std))
        else:
            averages.update(kid_mean=None, kid_std=None)
    result = {k: v for k, v in protocol.items() if k != 'samples'}
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ('torch','torchmetrics','torch-fidelity','pyiqa','lpips'):
        try: versions[package] = version(package)
        except PackageNotFoundError: versions[package] = None
    result.update(metrics=averages, samples=per_image, metric_protocol='native-RGB;alex-LPIPS;DiT-IC-original-patch-FID;KID-subset1000;' + json.dumps(versions,sort_keys=True),
                  fid_kid_note='FID omitted below 50 images; KID additionally requires >=1000 extracted patches. No substitute values.')
    atomic_json(result, output)
    return result

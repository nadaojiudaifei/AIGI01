"""Rate-distortion aggregation with overlapping-domain PCHIP BD-rate."""
from pathlib import Path
import json
import math
import numpy as np
from .utils import atomic_json, canonical_hash


def bd_rate(rate_a, score_a, rate_b, score_b, higher_is_better=True):
    from scipy.interpolate import PchipInterpolator
    if min(len(rate_a), len(rate_b)) < 4:
        raise ValueError('BD-rate requires at least four measured quality points per method')
    def prepare(rate, score):
        r, s = np.asarray(rate, dtype=float), np.asarray(score, dtype=float)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(s)) or np.any(r <= 0):
            raise ValueError('Non-finite scores or nonpositive rates')
        if not higher_is_better:
            s = -s
        idx = np.argsort(s)
        s, r = s[idx], r[idx]
        if np.any(np.diff(s) <= 0) or np.any(np.diff(r) <= 0):
            raise ValueError('Non-monotone/duplicate RD points; report raw points, not misleading BD-rate')
        return s, np.log(r)
    sa, ra = prepare(rate_a, score_a)
    sb, rb = prepare(rate_b, score_b)
    lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
    if hi <= lo:
        raise ValueError('No quality overlap; extrapolated BD-rate is not reported')
    delta = (PchipInterpolator(sb, rb).integrate(lo, hi) - PchipInterpolator(sa, ra).integrate(lo, hi)) / (hi-lo)
    return float(100 * np.expm1(delta))


def compare(files, output, metric='psnr', baseline='ditic', plot=False):
    rows = [json.loads(Path(p).read_text()) for p in files]
    signatures = {canonical_hash(sorted(r['image_sha256'] for r in record['samples'])) for record in rows}
    if len(signatures) != 1:
        raise ValueError('RD comparison requires exactly the same reference image set')
    if len({r['metric_protocol'] for r in rows}) != 1:
        raise ValueError('Cannot mix metric implementations/protocols')
    groups = {}
    for r in rows:
        groups.setdefault(r.get('experiment', r['method']), []).append(r)
    if baseline not in groups:
        raise ValueError('Missing baseline')
    base = groups[baseline]
    rb, sb = [r['bpp_total'] for r in base], [r['metrics'][metric] for r in base]
    scores = {}
    for name, group in groups.items():
        if name == baseline:
            continue
        rates, values = [r['bpp_total'] for r in group], [r['metrics'][metric] for r in group]
        scores[name] = bd_rate(rb, sb, rates, values, higher_is_better=metric in {'psnr', 'ms_ssim', 'musiq', 'clipiqa'})
    atomic_json(dict(baseline=baseline, metric=metric, bd_rate_percent=scores, rate='total bits / original pixels',
                     convention='negative means fewer bits over measured overlapping quality interval'), output)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, group in groups.items():
            points = sorted((r['bpp_total'], r['metrics'][metric]) for r in group)
            ax.plot([p[0] for p in points], [p[1] for p in points], marker='o', label=name)
        ax.set_xlabel('Total bits per original pixel')
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(Path(output).with_suffix('.png')), dpi=160)
        plt.close(fig)
    return scores

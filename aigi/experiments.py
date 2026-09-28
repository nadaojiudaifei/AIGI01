"""Explicit experiment plans. Generating a matrix never starts expensive training."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import itertools
import json
import os
import subprocess
import sys
import time
from .config import Components, save_config
from .utils import atomic_json, canonical_hash, sha256_file

IDEA1 = ['learned_token_weights', 'map_distillation', 'texture', 'importance', 'difficulty',
         'explanation', 'correction', 'text_entropy', 'waterfill', 'weighted_loss']
IDEA2 = ['snr_time', 'semantic_time', 'time_residual', 'local_adaln', 'dual_guidance']
COMMON = ['self_distillation', 'latent_prompt']
RATE_WEIGHTS = {1: 16., 2: 2., 3: .5, 4: .25}


def variants(base, ablations=True, add_one=True, factorial=None):
    result = []
    for method in ('ditic', 'cadc_style', 'idea1', 'idea2', 'joint'):
        c = deepcopy(base)
        c.method = method
        result.append((method, c))
    if ablations:
        for name in IDEA1 + IDEA2 + COMMON:
            c = deepcopy(base)
            c.method = 'joint'
            setattr(c.components, name, False)
            if name == 'map_distillation':
                c.maps_checkpoint = ''
            result.append(('joint_without_' + name, c))
        for method in ('ditic', 'idea1'):
            c = deepcopy(base)
            c.method = method
            c.components.variance_flow = False
            result.append((method + '_without_variance_flow', c))
        for method in ('ditic', 'joint'):
            for name in ('gan', 'dists'):
                c = deepcopy(base)
                c.method = method
                setattr(c, name + '_weight', 0.)
                result.append((method + '_without_' + name, c))
            for vae, dit in ((16, 16), (32, 32), (32, 64), (64, 128)):
                c = deepcopy(base)
                c.method, c.vae_rank, c.dit_rank = method, vae, dit
                result.append((f'{method}_rank_{vae}_{dit}', c))
            c = deepcopy(base)
            c.method, c.finetune = method, 'full'
            result.append((method + '_full_finetune', c))
            c = deepcopy(base)
            c.method, c.dit_from_scratch, c.finetune = method, True, 'full'
            result.append((method + '_scratch_dit', c))
    if add_one:
        for family, names in (('idea1', IDEA1), ('idea2', IDEA2)):
            for enabled in names:
                c = deepcopy(base)
                c.method = family
                for name in names:
                    setattr(c.components, name, name == enabled)
                prerequisites = {
                    'learned_token_weights': ['importance'],
                    'map_distillation': ['importance', 'difficulty', 'explanation', 'correction'],
                    'difficulty': ['correction'], 'explanation': ['correction'],
                    'waterfill': ['importance', 'difficulty'],
                    'weighted_loss': ['importance'], 'local_adaln': ['snr_time'],
                }
                for dependency in prerequisites.get(enabled, []):
                    setattr(c.components, dependency, True)
                # These are add-one tests within the corresponding new framework, not
                # a claim that its necessary backbone routing is identical to DiT-IC.
                result.append((family + '_only_' + enabled, c))
    if factorial:
        names = IDEA1 if factorial == 'idea1' else IDEA2
        for bits in itertools.product((False, True), repeat=len(names)):
            c = deepcopy(base)
            c.method = factorial
            for name, enabled in zip(names, bits):
                setattr(c.components, name, enabled)
            tag = ''.join('1' if b else '0' for b in bits)
            result.append((factorial + '_factorial_' + tag, c))
    return result


def build_matrix(base, output, qualities=(1, 2, 3, 4), ablations=True, add_one=True,
                 factorial=None, max_jobs=1000, gpus=1, datasets=(), protocol='main'):
    if protocol not in {'main','ablation256'} or gpus < 1:
        raise ValueError('Invalid protocol or GPU count')
    stages = [('stage1',256,100000,0.),('stage2',512,60000,.1)] if protocol=='main' else [('ablation256',256,60000,.1)]
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for quality in qualities:
        for name, cfg in variants(base, ablations, add_one, factorial):
            cfg.quality, cfg.rate_weight = quality, RATE_WEIGHTS[quality]
            cfg.experiment = name
            if not cfg.components.map_distillation:
                cfg.maps_checkpoint = ''
            # Explicit checkpoint filename substitution only for official merged qN names.
            old = f'q{base.quality}_merge_ema.pt'
            cfg.weights = cfg.weights.replace(old, f'q{quality}_merge_ema.pt')
            work = root / f'{name}_q{quality}'
            previous = None
            for stage, crop, steps, gan in stages:
                c = deepcopy(cfg)
                c.stage, c.crop, c.steps = 'codec', crop, steps
                c.output = str(work / stage)
                c.gan_weight = 0. if name.endswith('_without_gan') else gan
                c.init_checkpoint = str(previous) if previous else ''
                cpath = work / (stage + '.json')
                c.validate()
                save_config(c, cpath)
                command = ['torchrun', '--standalone', f'--nproc_per_node={gpus}', '-m', 'aigi', 'train', '--config', str(cpath)] if gpus > 1 else [sys.executable, '-m', 'aigi', 'train', '--config', str(cpath)]
                checkpoint = Path(c.output) / 'last.pt'
                jobs.append(dict(id=f'{name}_q{quality}_{stage}', kind='train', argv=command,
                                 config=str(cpath), expected=str(checkpoint), dependency=str(previous) if previous else None))
                previous = checkpoint
            for manifest in datasets:
                dname = Path(manifest).parent.name + '_' + Path(manifest).stem + '_' + canonical_hash(str(Path(manifest).resolve()))[:8]
                evaldir = work / ('eval_' + dname)
                jobs.append(dict(id=f'{name}_q{quality}_infer_{dname}', kind='infer', config=str(cpath),
                                 argv=[sys.executable, '-m', 'aigi', 'infer', '--config', str(cpath), '--checkpoint', str(previous),
                                       '--manifest', str(Path(manifest).resolve()), '--output', str(evaldir)],
                                 expected=str(evaldir / 'inference.json'), dependency=str(previous)))
                jobs.append(dict(id=f'{name}_q{quality}_metrics_{dname}', kind='evaluate',
                                 argv=[sys.executable, '-m', 'aigi', 'evaluate', '--input', str(evaldir / 'inference.json'),
                                       '--output', str(evaldir / 'metrics.json'), '--metrics',
                                       'psnr', 'ms_ssim', 'lpips', 'dists', 'niqe', 'clipiqa', 'musiq', 'fid', 'kid'],
                                 expected=str(evaldir / 'metrics.json'), dependency=str(evaldir / 'inference.json')))
    if len(jobs) > max_jobs:
        raise ValueError(f'{len(jobs)} planned jobs exceeds --max-jobs {max_jobs}; no training was started')
    atomic_json({'version': 1, 'jobs': jobs, 'count': len(jobs),
                 'protocol': protocol + ';'+base.augmentation+';views='+str(base.views_per_image)+';optimizer='+base.optimizer+';gan='+base.gan_mode+';FP32;fresh run per variant;explicit engineering quality grid'}, root / 'matrix.json')
    return str(root / 'matrix.json')


def run_matrix(path, execute=False, limit=0):
    matrix = json.loads(Path(path).read_text())
    root = Path(path).resolve().parent
    report = root / 'job_status.json'
    statuses = json.loads(report.read_text()) if report.exists() else {}
    memo={}
    def input_hash(path):
        p=Path(path).resolve()
        if not p.is_file():return None
        stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
        if key not in memo:memo[key]=sha256_file(p)
        return memo[key]
    def inputs(job):
        if not execute:return {}
        paths=[]
        if job.get('config'):
            config=json.loads(Path(job['config']).read_text())
            paths += [config[k] for k in ('train_manifest','val_manifest','weights','init_checkpoint','maps_checkpoint') if config.get(k)]
        argv=job['argv']
        for flag in ('--manifest','--questions','--input','--predictions','--a','--b'):
            if flag in argv:paths.append(argv[argv.index(flag)+1])
        return {str(Path(p).resolve()):input_hash(p) for p in paths}
    for index, job in enumerate(matrix['jobs']):
        if limit and index >= limit:
            break
        signature = canonical_hash(dict(job=job, config=Path(job['config']).read_text() if job.get('config') else None,
                                        dependency=sha256_file(job['dependency']) if execute and job.get('dependency') and Path(job['dependency']).is_file() else None,
                                        code=canonical_hash({p.name:input_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}), inputs=inputs(job)))
        print(json.dumps(job), flush=True)
        if not execute:
            continue
        prior = statuses.get(job['id'], {})
        if prior.get('signature') == signature and prior.get('returncode') == 0 and Path(job['expected']).is_file() and prior.get('output_sha256') == sha256_file(job['expected']):
            continue
        if job.get('dependency') and not Path(job['dependency']).is_file():
            raise FileNotFoundError('Job dependency missing: ' + job['dependency'])
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        begin = time.time()
        with (logs / (job['id'] + '.log')).open('w') as f:
            result = subprocess.run(job['argv'], stdout=f, stderr=subprocess.STDOUT, check=False)
        statuses[job['id']] = dict(signature=signature, returncode=result.returncode, seconds=time.time()-begin,
                                   output_sha256=sha256_file(job['expected']) if Path(job['expected']).is_file() else None)
        atomic_json(statuses, report)
        if result.returncode or not Path(job['expected']).is_file():
            raise RuntimeError('Experiment failed: ' + job['id'] + '; inspect logs. No failure was skipped.')
    return str(report)

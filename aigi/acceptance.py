"""Remote integration gate: actual weights, actual rANS, separate encoder/decoder processes.

This gate is intentionally not described as having passed until its report says so.
"""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
from .config import save_config
from .utils import atomic_json, jsonl_read, sha256_file


def acceptance(cfg, output, execute=False, device='cuda', extended=False, gpus=1):
    if gpus < 1:raise ValueError('gpus must be positive')
    if gpus > 1 and not extended:raise ValueError('Multiple-GPU acceptance requires --extended')
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    rows=list(jsonl_read(cfg.train_manifest))[:max(2,gpus)]
    val=list(jsonl_read(cfg.val_manifest))[:1]
    for items,source in ((rows,cfg.train_manifest),(val,cfg.val_manifest)):
        for row in items:
            if not Path(row['image']).is_absolute():
                row['image']=str((Path(source).resolve().parent/row['image']).resolve())
    if len(rows)<max(2,gpus) or not val: raise ValueError('Acceptance needs max(2,gpus) training images and a disjoint validation image')
    def manifest(items,path):
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
        return str(path)
    trainpath=manifest(rows,root/'train.jsonl'); valpath=manifest(val,root/'val.jsonl')
    jobs=[]
    common=deepcopy(cfg)
    common.crop=256; common.quality=cfg.quality; common.train_manifest=trainpath; common.val_manifest=valpath
    common.teacher_cache=str(root/'teacher'); common.stage='codec'; common.batch_size=1; common.workers=0
    common.accumulation=1; common.steps=2; common.save_every=1; common.val_every=1
    common.augmentation='center'; common.views_per_image=1
    common.gan_weight=0.; common.lpips_weight=0.; common.dists_weight=0.; common.init_checkpoint=''; common.maps_checkpoint=''
    cpath=root/'teacher.json'; save_config(common,cpath)
    for source in (trainpath,valpath):
        jobs.append([sys.executable,'-m','aigi','teacher','--config',str(cpath),'--manifest',source,'--device',device])
    row=val[0]; prompt=root/'prompt.txt'; prompt.write_text(row.get('prompt',''),encoding='utf-8')
    checks=[]
    for method in ('ditic','cadc_style','idea1','idea2','joint'):
        c=deepcopy(common); c.method=method; c.experiment='acceptance_'+method; c.output=str(root/method/'train')
        path=root/(method+'.json'); save_config(c,path)
        ckpt=str(Path(c.output)/'last.pt'); packet=root/method/'sample.aigi'
        jobs.append([sys.executable,'-m','aigi','train','--config',str(path),'--device',device])
        jobs.append([sys.executable,'-m','aigi','encode','--config',str(path),'--checkpoint',ckpt,'--input',row['image'],
                     '--prompt-file',str(prompt),'--output',str(packet),'--device',device])
        for copy in ('decoded_a.png','decoded_b.png'):
            jobs.append([sys.executable,'-m','aigi','decode','--config',str(path),'--checkpoint',ckpt,
                         '--input',str(packet),'--output',str(root/method/copy),'--device',device])
        checks.append((method,root/method/'decoded_a.png',root/method/'decoded_b.png'))
    # Interrupt at step 1 while preserving the original two-step LR horizon;
    # compare against the uninterrupted joint run, not against a changed config.
    resume_cfg=deepcopy(common);resume_cfg.method='joint';resume_cfg.experiment='acceptance_joint'
    resume_cfg.output=str(root/'joint_resumed/train');rp=root/'joint_resumed.json';save_config(resume_cfg,rp)
    rc=str(Path(resume_cfg.output)/'last.pt')
    jobs += [[sys.executable,'-m','aigi','train','--config',str(rp),'--stop-after','1','--device',device],
             [sys.executable,'-m','aigi','train','--config',str(rp),'--resume',rc,'--device',device],
             [sys.executable,'-m','aigi','checkpoint-compare','--a',str(root/'joint/train/last.pt'),
              '--b',rc,'--output',str(root/'resume_equivalence.json')]]
    if extended:
        gan=deepcopy(common);gan.method='joint';gan.gan_weight=.1;gan.gan_start_fraction=0.
        gan.accumulation=2;gan.output=str(root/'gan/train');gp=root/'gan.json';save_config(gan,gp)
        jobs.append([sys.executable,'-m','aigi','train','--config',str(gp),'--device',device])
        jobs.append([sys.executable,'-m','aigi','infer','--config',str(root/'joint.json'),
                     '--checkpoint',str(root/'joint/train/last.pt'),'--manifest',valpath,
                     '--output',str(root/'metrics_images'),'--device',device])
        jobs.append([sys.executable,'-m','aigi','evaluate','--input',str(root/'metrics_images/inference.json'),
                     '--output',str(root/'metrics.json'),'--device',device])
        if gpus>1:
            ddp=deepcopy(common);ddp.method='joint';ddp.output=str(root/'ddp/train')
            dp=root/'ddp.json';save_config(ddp,dp)
            jobs.append(['torchrun','--standalone',f'--nproc_per_node={gpus}','-m','aigi','train','--config',str(dp)])
    atomic_json({'jobs':jobs,'executed':False,'tests':'two optimization steps per method; real entropy encoding; fresh-process repeated decoding'},root/'plan.json')
    if not execute:
        print(root/'plan.json'); return
    statuses=[]
    atomic_json({'jobs':[],'passed':False,'phase':'starting'},root/'acceptance.json')
    for i,argv in enumerate(jobs):
        with (root/f'job_{i:03d}.log').open('w') as f:
            result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=False)
        statuses.append({'argv':argv,'returncode':result.returncode})
        atomic_json({'jobs':statuses,'passed':False},root/'acceptance.json')
        if result.returncode: raise RuntimeError(f'Acceptance job {i} failed; inspect job_{i:03d}.log')
    for method,a,b in checks:
        if sha256_file(a)!=sha256_file(b): raise AssertionError('Fresh-process decoding differs: '+method)
    atomic_json({'jobs':statuses,'passed':True,'methods':[m for m,_,_ in checks],
                 'extended_requested':extended,'ddp_requested':extended and gpus>1,
                 'limits':'Finite smoke gate only. FID/KID remain unavailable on this small sample; no convergence or scientific improvement claim. DDP smoke is not an equivalence proof.'},root/'acceptance.json')

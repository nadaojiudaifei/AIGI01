"""Standard-library diagnostics first: even a missing/broken torch gets a report."""
from importlib import metadata
from pathlib import Path
import json
import os
import platform
import sys
import tempfile

REQUIRED={'torch':'2.8.0','torchvision':'0.23.0','compressai':'1.2.8','diffusers':'0.35.2',
          'transformers':'4.55.4','peft':'0.17.1','huggingface-hub':'0.34.4','datasets':'4.0.0',
          'lpips':'0.1.4','pyiqa':'0.1.15.post2','torchmetrics':'1.8.2','torch-fidelity':'0.3.0','numpy':'1.26.4'}


def doctor(config=None,output='aigi_runs/doctor.json',require_gpu=False):
    report={'python':sys.version,'platform':platform.platform(),'python_target':'3.12','packages':{},'errors':[],
            'note':'Read-only preflight; not a training or entropy round-trip validation.'}
    if sys.version_info[:2]!=(3,12):report['errors'].append('Target Python is 3.12')
    for name,expected in REQUIRED.items():
        try:installed=metadata.version(name)
        except metadata.PackageNotFoundError:installed=None
        matched=installed is not None and installed.split('+')[0]==expected
        report['packages'][name]={'installed':installed,'target':expected,'matches':matched}
        if not matched:report['errors'].append(f'{name}: target {expected}, installed {installed}')
    try:
        import torch
        report.update(cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,gpu_count=torch.cuda.device_count())
    except Exception as exc:
        report.update(cuda_available=False,cuda_build=None,gpu_count=0)
        report['errors'].append('torch import failed: '+str(exc))
    if require_gpu and not report['cuda_available']:report['errors'].append('No CUDA GPU')
    if config:
        try:
            from .config import load_config
            c=load_config(config)
            paths={'upstream':Path(c.upstream_root)/'models/DiT_IC.py','config':Path(c.upstream_root)/c.upstream_config,
                   'weights':Path(c.weights),'sana':Path(c.sana_path),'train_manifest':Path(c.train_manifest),
                   'val_manifest':Path(c.val_manifest)}
            report['paths']={k:{'path':str(p.resolve()),'exists':p.exists()} for k,p in paths.items()}
            for k,p in paths.items():
                if not p.exists():report['errors'].append('Missing '+k+': '+str(p))
            report['config']=c.to_dict()
        except Exception as exc:report['errors'].append('Configuration preflight failed: '+str(exc))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,delete=False) as f:
        json.dump(report,f,indent=2);temp=f.name
    os.replace(temp,output);print(json.dumps(report,indent=2))
    return 1 if report['errors'] else 0

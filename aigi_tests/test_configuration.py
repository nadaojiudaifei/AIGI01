import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from aigi.config import Config,Components,save_config,load_config
from aigi.__main__ import main,parser
from aigi.experiments import variants,build_matrix,run_matrix,IDEA1,IDEA2
from aigi.checkpoint import rng_state,restore_rng,save_training,read_checkpoint,EMA
from aigi.compare import bd_rate
from aigi.utils import atomic_json

def test_config_roundtrip(tmp_path):
    c=Config();p=tmp_path/'c.json';save_config(c,p);assert load_config(p)==c
    p.write_text('{"typo":true}')
    with pytest.raises(TypeError):load_config(p)

@pytest.mark.parametrize('kw',[dict(crop=128),dict(batch_size=0),dict(quality=0),dict(method='bogus'),dict(dit_from_scratch=True),dict(lpips_weight=-1),dict(offline='false')])
def test_bad_configs(kw):
    with pytest.raises((ValueError,TypeError)):Config(**kw).validate()

def test_flag_type():
    with pytest.raises(TypeError):Config(components=Components(texture='false')).validate()

def test_variants_valid_unique():
    vs=variants(Config());names=[n for n,_ in vs];assert len(names)==len(set(names))
    for n,c in vs:c.validate()
    table=dict(vs);assert table['idea1_only_difficulty'].components.correction
    assert not table['joint_without_text_entropy'].components.text_entropy
    assert table['ditic_scratch_dit'].finetune=='full'
    assert 'joint_without_variance_flow' not in table

def test_factorial_counts():
    assert len(variants(Config(),False,False,'idea2'))==37
    assert len(variants(Config(),False,False,'idea1'))==1029

def test_matrix_dataset_collision_and_dryrun(tmp_path):
    paths=[tmp_path/'natural'/'manifest.jsonl',tmp_path/'aigi'/'manifest.jsonl']
    p=build_matrix(Config(),tmp_path/'plan',[3],False,False,datasets=paths)
    jobs=json.loads(Path(p).read_text())['jobs'];assert len(jobs)==30
    assert len({j['id'] for j in jobs})==30
    run_matrix(p,False);assert not (tmp_path/'plan/job_status.json').exists()

def test_matrix_limit(tmp_path):
    with pytest.raises(ValueError):build_matrix(Config(),tmp_path/'p',[3],False,False,max_jobs=1)

def test_configure_cli(tmp_path):
    p=tmp_path/'c.json';assert main(['configure','--output',str(p),'--method','idea1','--set','components.texture=false'])==0
    assert load_config(p).components.texture is False
    with pytest.raises(ValueError):main(['configure','--output',str(p),'--set','typo=1'])

def test_cli_catalog(capsys):
    assert main(['catalog'])==0
    assert 'diffusiondb' in capsys.readouterr().out

def test_checkpoint_rng_resume(tmp_path):
    s=rng_state();expected=(random.random(),np.random.rand(),torch.rand(2))
    restore_rng(s);assert random.random()==expected[0] and np.random.rand()==expected[1] and torch.equal(torch.rand(2),expected[2])
    m=torch.nn.Linear(3,2);opt=torch.optim.Adam(m.parameters());p=tmp_path/'last.pt'
    save_training(p,m,Config(),5,opt,None,ema={},rank_rng=[s]);ck=read_checkpoint(p)
    assert ck['step']==5 and ck['config']['method']=='joint'
    n=torch.nn.Linear(3,2);n.load_state_dict(ck['model']);assert torch.equal(m.weight,n.weight)

def test_ema():
    m=torch.nn.Linear(2,2,bias=False);e=EMA(m,.5);old=m.weight.detach().clone()
    with torch.no_grad():m.weight.add_(2)
    e.update(m);e.apply(m);assert torch.allclose(m.weight,old+1)

def test_bd_rate_exact_savings():
    r=[.01,.02,.04,.08];s=[20,22,24,26]
    assert bd_rate(r,s,[x*.8 for x in r],s)==pytest.approx(-20.)
    assert bd_rate(r,s,r,s)==pytest.approx(0.)
    with pytest.raises(ValueError):bd_rate(r,s,r,[40,42,44,46])
    with pytest.raises(ValueError):bd_rate(r[:2],s[:2],r,s)

def test_atomic_json_reject_nan(tmp_path):
    with pytest.raises(ValueError):atomic_json({'x':float('nan')},tmp_path/'x.json')

def test_bootstrap_snapshot_integrity(tmp_path):
    path=Path(__file__).parents[1]/'aigi_scripts/bootstrap_upstream.py';spec=importlib.util.spec_from_file_location('bootstrap',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    root=tmp_path/'repo';root.mkdir();subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    (root/'original.py').write_text('x=1\n');subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=test','-c','user.email=test@localhost','commit','-m','test'],check=True,capture_output=True)
    mod.PIN=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    lock=mod.upstream_lock(root);(root/'aigi_docs').mkdir();(root/'aigi_docs/UPSTREAM_LOCK.json').write_text(json.dumps(lock));assert mod.verify(root)==1
    (root/'original.py').write_text('x=2\n')
    with pytest.raises(RuntimeError):mod.verify(root)

def test_ablation_protocol(tmp_path):
    p=build_matrix(Config(),tmp_path/'plan',[3],False,False,protocol='ablation256')
    jobs=json.loads(Path(p).read_text())['jobs'];assert len(jobs)==5
    for job in jobs:
        c=load_config(job['config']);assert c.crop==256 and c.steps==60000

def test_hf_preload_reject_shadow(monkeypatch):
    import types
    from aigi.utils import preload_hf_datasets
    monkeypatch.setitem(sys.modules,'datasets',types.ModuleType('datasets'))
    with pytest.raises(RuntimeError):preload_hf_datasets()

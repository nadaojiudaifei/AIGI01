import csv
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from aigi.config import Config
from aigi.augmentation import view_spec, apply_view
from aigi.data import record_image, ManifestDataset, cache_key
from aigi.optimization import learning_rate, make_optimizer, adversarial_coefficient
from aigi.ocrbench import prepare, resolve_inputs, annotations, score, model_fingerprint
from aigi.study import build_study, aggregate_study, cluster_statistics
from aigi.instrumentation import instrument, ModuleRecorder
from aigi.suite import build_suite
from aigi.utils import sha256_file


def fixture_image(tmp_path,name='image.png',value=110):
    p=tmp_path/name;Image.new('RGB',(320,288),(value,90,20)).save(p)
    return record_image(p,tmp_path,'fixture')


def test_augmentation_is_shared_deterministic_view(tmp_path):
    r=fixture_image(tmp_path)
    s=view_spec(r,256,3,42,'cached_random')
    assert s==view_spec(r,256,3,42,'cached_random')
    assert s!=view_spec(r,256,2,42,'cached_random')
    assert apply_view(Image.open(r['image']).convert('RGB'),s).shape==(3,256,256)


def test_center_preserves_existing_cache_key(tmp_path):
    r=fixture_image(tmp_path);p=tmp_path/'m.jsonl';p.write_text(json.dumps(r)+'\n')
    ds=ManifestDataset(p,256);assert ds.key_at(0)==cache_key(r,256,3)
    ds2=ManifestDataset(p,256,augmentation='cached_random',views_per_image=4)
    assert len(ds2)==4 and len({ds2.key_at(i) for i in range(4)})==4
    assert ds2.key_at(0)!=ds.key_at(0)


@pytest.mark.parametrize('name,value',[('views_per_image',1.5),('batch_size',True),('quality',2.5),('hidden',12.2),('dit_rank',0),('lr',float('nan')),('rate_weight',float('inf')),('weight_decay',-.1)])
def test_invalid_config_numeric_rejected(name,value):
    c=Config();setattr(c,name,value)
    with pytest.raises((ValueError,TypeError)):c.validate()


@pytest.mark.parametrize('step,ratio',[(0,1),(49999,1),(50000,.5),(80000,.25),(90000,.125),(100000,.125)])
def test_paper_fraction_lr(step,ratio):
    c=Config();assert learning_rate(c,step)==pytest.approx(c.lr*ratio)


def test_small_step_schedule_no_initial_decay():
    c=Config(steps=2);assert learning_rate(c,0)==c.lr


def test_explicit_adamw_and_detached_adaptive_gan():
    c=Config(gan_weight=.1);p=torch.nn.Parameter(torch.tensor(2.))
    assert isinstance(make_optimizer([p],c),torch.optim.AdamW)
    value=adversarial_coefficient(p.square(),3*p,p,c)
    assert not value.requires_grad and value.item()==pytest.approx(.1*4/3.0001)
    c.gan_mode='fixed';assert adversarial_coefficient(p,p,p,c).item()==pytest.approx(.1)


def qa_rows(image_name):
    return [{'id':0,'dataset_name':'fixture','type':'text recognition en','image_path':image_name,
             'question':'PRIVATE_QA_QUESTION','answers':['PRIVATE_QA_ANSWER']}]


def test_ocr_prepare_does_not_leak_qa_to_codec(tmp_path):
    r=fixture_image(tmp_path);q=tmp_path/'qa.json';q.write_text(json.dumps(qa_rows('image.png')))
    manifest=Path(prepare(q,tmp_path,tmp_path/'prepared'))
    text=manifest.read_text();assert 'PRIVATE_QA' not in text
    assert json.loads(text)['prompt']==''
    questions=annotations(tmp_path/'prepared/questions.json');assert questions[0]['question']=='PRIVATE_QA_QUESTION'
    assert resolve_inputs(questions)[0][1]==r['sha256']


def test_ocr_deduplicates_identical_pixels(tmp_path):
    fixture_image(tmp_path);fixture_image(tmp_path,'duplicate.png')
    q=tmp_path/'qa.json';rows=qa_rows('image.png')+qa_rows('duplicate.png');rows[1]['id']=1;q.write_text(json.dumps(rows))
    m=Path(prepare(q,tmp_path,tmp_path/'prepared'))
    assert len(m.read_text().splitlines())==1
    assert len(annotations(tmp_path/'prepared/questions.json'))==2


def test_ocr_identity_duplicate_and_escape_rejected(tmp_path):
    r=qa_rows('../outside.png');q=tmp_path/'q.json';q.write_text(json.dumps(r+r))
    with pytest.raises(ValueError,match='Duplicate'):annotations(q)
    q.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='escapes'):prepare(q,tmp_path,tmp_path/'prepared')


def test_ocr_resolve_requires_exact_reference_set(tmp_path):
    fixture_image(tmp_path);q=tmp_path/'q.json';q.write_text(json.dumps(qa_rows('image.png')))
    prepare(q,tmp_path,tmp_path/'prepared');rows=annotations(tmp_path/'prepared/questions.json')
    inf=tmp_path/'inf.json';inf.write_text(json.dumps({'samples':[]}))
    with pytest.raises(ValueError,match='exactly match'):resolve_inputs(rows,inf)
    Path(rows[0]['_aigi_original_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):resolve_inputs(rows)


def test_ocr_score_plan_uses_official_not_surrogate(tmp_path):
    rows=qa_rows('x.png');rows[0]['predict']='text';p=tmp_path/'p.json';p.write_text(json.dumps(rows))
    plan=score(p,tmp_path/'official',tmp_path/'scores',execute=False)
    assert plan['commands'][0][1].endswith('OCRBench_v2/eval_scripts/eval.py')
    assert not (tmp_path/'scores/scored.json').exists()


def study_fixture(tmp_path):
    r=fixture_image(tmp_path);inputs=[]
    for label in ('ditic','joint'):
        p=tmp_path/(label+'.json')
        p.write_text(json.dumps({'method':label,'samples':[{'id':'x','image_sha256':r['sha256'],
                        'original':r['image'],'reconstruction':r['image'],'bpp_total':.1}]}));inputs.append(p)
    build_study(*inputs,tmp_path/'study')
    return tmp_path/'study/PRIVATE_KEY.json'


def responses(path,key,n=3):
    d=json.loads(key.read_text())
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['participant','study_id','trial','choice','criterion']);w.writeheader()
        for i in range(n):
            w.writerow({'participant':f'p{i}','study_id':d['study_id'],'trial':'0','choice':'Tie','criterion':'realism'})


def test_blind_study_keeps_key_out_of_public(tmp_path):
    key=study_fixture(tmp_path);page=(key.parent/'public/index.html').read_text()
    assert 'ditic' not in page and 'joint' not in page and 'reference.png' not in page
    assert not (key.parent/'public/PRIVATE_KEY.json').exists()


def test_study_real_response_aggregation_and_duplicates(tmp_path):
    key=study_fixture(tmp_path);p=tmp_path/'answers.csv';responses(p,key)
    r=aggregate_study(key,[p],tmp_path/'result.json',draws=100)
    assert r['participants']==3 and r['counts']['tie']==3
    assert r['first_method_statistics']['preference_including_half_ties']==.5
    with pytest.raises(ValueError,match='Duplicate'):aggregate_study(key,[p,p],tmp_path/'bad.json',draws=100)


def test_study_no_participants_is_not_success(tmp_path):
    key=study_fixture(tmp_path);p=tmp_path/'empty.csv';responses(p,key,n=0)
    with pytest.raises(ValueError,match='No participant'):aggregate_study(key,[p],tmp_path/'result.json',draws=100)


def test_cluster_statistics_exact_participant_test():
    r=cluster_statistics([[1,1],[1,1],[1,1]],draws=100)
    assert r['participant_sign_flip_p_two_sided']==pytest.approx(.25)
    assert r['participant_cluster_ci95']==[1.,1.]
    r=cluster_statistics([[.5,1]],draws=100)
    assert r['participant_cluster_ci95'] is None


def test_flops_are_real_and_not_claimed_complete():
    m=torch.nn.Linear(3,2,bias=False)
    with torch.no_grad():result,report=instrument(lambda:m(torch.ones(4,3)),{'linear':m})
    assert result.shape==(4,2) and report['registered_flops']==48
    assert report['modules']['linear']['calls']==1 and not report['flops_complete']
    assert not m._forward_hooks and not m._forward_pre_hooks


def test_hook_cleanup_on_failure():
    from torch.utils.flop_counter import FlopCounterMode
    m=torch.nn.Linear(3,2);c=FlopCounterMode(display=False)
    with pytest.raises(RuntimeError):
        with c,ModuleRecorder({'linear':m},c):m(torch.ones(2,4))
    assert not m._forward_hooks and not m._forward_pre_hooks


def test_suite_chooses_terminal_training_stage(tmp_path):
    source={'jobs':[{'id':'joint_q3_stage1','kind':'train','config':'c1.json','expected':'s1.pt'},
                    {'id':'joint_q3_stage2','kind':'train','config':'c2.json','expected':'s2.pt'}]}
    p=tmp_path/'matrix.json';p.write_text(json.dumps(source));out=build_suite(p,tmp_path/'suite')
    jobs=json.loads(Path(out).read_text())['jobs'];assert len(jobs)==1 and jobs[0]['dependency']=='s2.pt'
    with pytest.raises(ValueError,match='requires'):build_suite(p,tmp_path/'bad',questions='q.json')


def test_checkpoint_resume_exact_comparator(tmp_path):
    from aigi.checkpoint import compare_training
    state={'format':'aigi01-training-v1','step':2,'model':{'w':torch.tensor([1.,2.])}}
    a=tmp_path/'a.pt';b=tmp_path/'b.pt';torch.save(state,a);torch.save(state,b)
    assert compare_training(a,b,tmp_path/'equal.json')['passed']
    state['model']['w'][0]=3;torch.save(state,b)
    with pytest.raises(ValueError,match='model.w'):compare_training(a,b,tmp_path/'bad.json')


def test_baseline_does_not_allocate_new_method_networks():
    from test_causal_loop import FakeBackbone
    from aigi.model import AIGICodec
    m=AIGICodec(Config(method='ditic',hidden=8),'cpu',backbone=FakeBackbone())
    assert m.entropy is None and m.rho is None and m.adaln is None
    assert all(name.startswith('backbone.') for name in m.state_dict())


def test_doctor_reports_missing_torch_without_crash(tmp_path,monkeypatch):
    import builtins
    from aigi.doctor import doctor
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name=='torch':raise ImportError('test missing runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    p=tmp_path/'doctor.json';assert doctor(output=p,require_gpu=True)==1
    r=json.loads(p.read_text());assert not r['cuda_available']
    assert any('torch import failed' in x for x in r['errors'])


def test_all_cli_help_without_runtime_imports():
    from aigi.__main__ import parser
    commands=parser()._subparsers._group_actions[0].choices
    for command in commands:
        with pytest.raises(SystemExit) as result:parser().parse_args([command,'--help'])
        assert result.value.code==0

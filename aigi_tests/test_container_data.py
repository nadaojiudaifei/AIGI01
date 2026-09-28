import hashlib
import json
import struct
import zlib
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from aigi.bitstream import pack,unpack,rate_report,MAGIC
from aigi.data import build_local_manifest,split_manifests,assert_disjoint,ManifestDataset,collate,cache_key
from aigi.utils import jsonl_read

@pytest.fixture
def meta():
    return dict(method='joint',quality=3,height=270,width=301,padded_height=512,padded_width=512,
                shape=[2,2],model_sha256='a'*64,text_sha256='b'*64,runtime='unit-tests-only')

@pytest.mark.parametrize('prompt',['','a brown dog','\u68d5\u8272\u7684\u72d7\U0001f436'])
def test_container_roundtrip(meta,prompt):
    raw=pack(meta,prompt,[[b'main'],[b'hyper']]);p=unpack(raw)
    assert p.prompt==prompt and p.strings==[[b'main'],[b'hyper']]
    r=rate_report(raw);assert abs(r['bpp_total']-sum(r[k] for k in ('bpp_entropy_payload','bpp_prompt','bpp_framing')))<1e-12
    assert r['pixels']==270*301

def test_external_prompt_and_rate(meta):
    raw=pack(meta,'secret',[[b'y'],[b'z']],True)
    assert unpack(raw,'secret').prompt=='secret';assert rate_report(raw)['bpp_prompt']==0
    with pytest.raises(ValueError):unpack(raw)
    with pytest.raises(ValueError):unpack(raw,'wrong')

@pytest.mark.parametrize('which',['flip','truncate','trailing','magic'])
def test_corruption_rejected(meta,which):
    raw=pack(meta,'p',[[b'y'],[b'z']])
    if which=='flip':raw=raw[:20]+bytes([raw[20]^1])+raw[21:]
    if which=='truncate':raw=raw[:-1]
    if which=='trailing':raw+=b'extra'
    if which=='magic':raw=b'notmagic'+raw[8:]
    with pytest.raises(ValueError):unpack(raw)
    with pytest.raises(ValueError):rate_report(raw)

@pytest.mark.parametrize('key,value',[('height',0),('quality',True),('shape',[1,1]),('padded_width',1024),('method','unknown')])
def test_bad_header(meta,key,value):
    meta[key]=value
    with pytest.raises(ValueError):pack(meta,'p',[[b'y'],[b'z']])

def test_prompt_bomb_and_schema(meta):
    with pytest.raises(ValueError):pack(meta,'x'*65537,[[b'y'],[b'z']])
    raw=pack(meta,'p',[[b'y'],[b'z']]);n=struct.unpack('>I',raw[8:12])[0];h=json.loads(raw[12:12+n]);h['extra']=1
    header=json.dumps(h).encode();body=MAGIC+struct.pack('>I',len(header))+header+raw[12+n:-32]
    with pytest.raises(ValueError):unpack(body+hashlib.sha256(body).digest())

def test_manifest_native_crop_hash(tmp_path):
    d=tmp_path/'images';d.mkdir();Image.new('RGB',(30,20),(20,30,40)).save(d/'a.png');(d/'a.txt').write_text('dog')
    path=tmp_path/'manifest.jsonl';assert build_local_manifest(d,path,kind='aigi',expected=1)==1
    ds=ManifestDataset(path,crop=256);item=ds[0];assert item['image'].shape==(3,256,256) and item['prompt']=='dog'
    assert collate([item,item])['image'].shape==(2,3,256,256)
    Image.new('RGB',(30,20)).save(d/'a.png')
    with pytest.raises(ValueError):ManifestDataset(path)[0]
    with pytest.raises(ValueError):build_local_manifest(d,path,expected=2)

def test_transitive_split_leakage(tmp_path):
    rows=[dict(image='a.png',pixel_sha256='x',prompt='alpha',prompt_origin='original',source_split='train'),
          dict(image='b.png',pixel_sha256='x',prompt='beta',prompt_origin='original',source_split='train'),
          dict(image='c.png',pixel_sha256='z',prompt='beta',prompt_origin='original',source_split='test')]
    p=tmp_path/'all.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    counts=split_manifests([p],tmp_path/'split');assert counts=={'train':0,'val':0,'test':2}

def test_official_validation_preserved(tmp_path):
    rows=[dict(image='a.png',pixel_sha256='x',prompt='A DOG',prompt_origin='original',source_split='train'),
          dict(image='b.png',pixel_sha256='y',prompt='a dog',prompt_origin='original',source_split='validation')]
    p=tmp_path/'all.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows));c=split_manifests([p],tmp_path/'s');assert c['val']==2

def test_cache_contract_required(tmp_path):
    d=tmp_path/'im';d.mkdir();Image.new('RGB',(16,16)).save(d/'x.png');p=tmp_path/'a.jsonl';build_local_manifest(d,p)
    with pytest.raises(FileNotFoundError):ManifestDataset(p,cache=tmp_path/'cache',require_cache=True)

#!/usr/bin/env python3
"""Import pristine pinned upstream, optionally push it FIRST, then add only new paths.

Standard library + git only. Run outside the destination. Never uses force-push.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PIN='cd43f5d9761fb34f5224622145629d3ff2b89ca1'
UPSTREAM='https://github.com/Eric-qi/DiT-IC.git'
DEFAULT_REMOTE='https://github.com/nadaojiudaifei/AIGI01.git'
ADDITIONS=['aigi','aigi_configs','aigi_scripts','aigi_tests','aigi_docs',
           'AIGI_README.md','aigi_requirements.txt','aigi_environment.yml','aigi_pytest.ini']


def git(root,*args,capture=False):
    command=['git','-C',str(root),*args]
    if capture: return subprocess.check_output(command,text=True).strip()
    subprocess.run(command,check=True)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def upstream_lock(root):
    names=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only','-z',PIN]).decode().split('\0')
    rows={}
    for name in filter(None,names):
        p=root/name
        rows[name]={'sha256':digest(p) if not p.is_symlink() else hashlib.sha256(os.readlink(p).encode()).hexdigest(),
                    'symlink':p.is_symlink()}
    return {'repository':UPSTREAM,'commit':PIN,'files':rows}


def verify(root):
    lock=json.loads((root/'aigi_docs/UPSTREAM_LOCK.json').read_text())
    bad=[]
    for name,record in lock['files'].items():
        p=root/name
        actual=(hashlib.sha256(os.readlink(p).encode()).hexdigest() if p.is_symlink() else digest(p)) if p.exists() else None
        if actual!=record['sha256'] or p.is_symlink()!=record['symlink']: bad.append(name)
    if bad: raise RuntimeError('Original upstream files changed/missing: '+', '.join(bad))
    return len(lock['files'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',required=True);p.add_argument('--remote',default=DEFAULT_REMOTE)
    p.add_argument('--overlay',default=str(Path(__file__).resolve().parent.parent));p.add_argument('--execute',action='store_true')
    p.add_argument('--push',action='store_true');p.add_argument('--author-name',default='AIGI01 bootstrap');p.add_argument('--author-email',default='bootstrap@localhost')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args(argv)
    root=Path(a.destination).expanduser().resolve(); overlay=Path(a.overlay).resolve()
    if a.verify_only: print(f'Verified {verify(root)} unchanged upstream paths');return
    print(json.dumps({'destination':str(root),'upstream':UPSTREAM,'commit':PIN,'remote':a.remote,
                      'sequence':['checkout pristine upstream','push upstream first if requested','add overlay without collisions','commit and push additions if requested']},indent=2))
    if not a.execute:return
    if root==overlay or root in overlay.parents or overlay in root.parents: raise ValueError('Destination and overlay must be separate, non-nested directories')
    if root.exists() and any(root.iterdir()): raise FileExistsError('Destination must be absent or empty; existing projects are never overwritten')
    if a.push:
        existing=subprocess.check_output(['git','ls-remote',a.remote],text=True)
        if existing.strip(): raise RuntimeError('Remote is not empty. Refusing to overwrite or force-push.')
    root.mkdir(parents=True,exist_ok=True)
    git(root,'init');git(root,'fetch','--no-tags',UPSTREAM,PIN)
    if git(root,'rev-parse','FETCH_HEAD',capture=True)!=PIN:raise RuntimeError('Upstream pin mismatch')
    git(root,'checkout','-b','main',PIN);git(root,'remote','add','origin',a.remote)
    lock=upstream_lock(root)
    if a.push:git(root,'push','-u','origin','main')
    # Only after the first push succeeds do we add the extension.
    for name in ADDITIONS:
        source=overlay/name
        if not source.exists():raise FileNotFoundError('Incomplete overlay: '+str(source))
        if (root/name).exists():raise FileExistsError('Upstream collision: '+name)
    for name in ADDITIONS:
        source=overlay/name; destination=root/name
        if source.is_dir():shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
        else:shutil.copy2(source,destination)
    (root/'aigi_docs/UPSTREAM_LOCK.json').write_text(json.dumps(lock,indent=2))
    # Local ignore file is not an upstream tracked project file.
    with (root/'.git/info/exclude').open('a') as f:
        f.write('\naigi_assets/\naigi_data/\naigi_cache/\naigi_runs/\n__pycache__/\n.pytest_cache/\n*.pyc\n.env\n')
    print(f'Verified {verify(root)} unchanged upstream paths')
    git(root,'add','--',*ADDITIONS)
    git(root,'-c','user.name='+a.author_name,'-c','user.email='+a.author_email,'commit','-m','feat: add isolated AIGI01 semantic compression research implementation')
    if a.push:git(root,'push','origin','main')
    print('Completed:',root)


if __name__=='__main__':main()

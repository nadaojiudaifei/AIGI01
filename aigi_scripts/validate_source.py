#!/usr/bin/env python3
"""Standard-library-only source checks. Does not install/import neural packages."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def validate(root, require_upstream=False):
    root=Path(root).resolve();python_files=[];configs=[];shells=[]
    for directory in ('aigi','aigi_tests','aigi_scripts'):
        for p in sorted((root/directory).rglob('*.py')):
            ast.parse(p.read_text(encoding='utf-8'),filename=str(p));python_files.append(str(p.relative_to(root)))
    if not python_files:raise ValueError('No extension Python sources')
    for p in sorted((root/'aigi_scripts').glob('*.sh')):
        subprocess.run(['bash','-n',str(p)],check=True);shells.append(str(p.relative_to(root)))
    for p in sorted((root/'aigi_configs').glob('*.json')):
        c=json.loads(p.read_text())
        if c.get('method') not in {'ditic','cadc_style','idea1','idea2','joint'}:raise ValueError('Bad method: '+str(p))
        if any(type(v) is not bool for v in c.get('components',{}).values()):raise ValueError('Nonboolean component')
        configs.append(str(p.relative_to(root)))
    count=None;lockpath=root/'aigi_docs/UPSTREAM_LOCK.json'
    if lockpath.exists():
        lock=json.loads(lockpath.read_text());count=0
        if lock['commit']!='cd43f5d9761fb34f5224622145629d3ff2b89ca1':raise ValueError('Unexpected upstream pin')
        for name,entry in lock['files'].items():
            path=root/name
            if not path.resolve().is_relative_to(root):raise ValueError('Unsafe upstream lock path')
            data=os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=entry['sha256'] or path.is_symlink()!=entry['symlink']:
                raise ValueError('Original upstream file changed: '+name)
            count+=1
    elif require_upstream:raise FileNotFoundError(lockpath)
    report={'passed':True,'python':platform.python_version(),'python_sources':python_files,
            'shell_scripts':shells,'configs':configs,'upstream_files_verified':count,
            'runtime_tests_executed':False,'gpu_tests_executed':False,
            'scope':'AST + JSON syntax/config basics + Bash syntax + pristine upstream bytes, not neural runtime validation'}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.');p.add_argument('--output',default='aigi_runs/source_validation.json')
    p.add_argument('--require-upstream',action='store_true');a=p.parse_args()
    report=validate(a.root,a.require_upstream);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'python_sources':len(report['python_sources']),
                      'shell_scripts':len(report['shell_scripts']),'configs':len(report['configs']),
                      'upstream_files_verified':report['upstream_files_verified']}))


if __name__=='__main__':main()

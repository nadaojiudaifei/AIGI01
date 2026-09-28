"""OCRBench V2: unique-image coding, fixed local VLM, official scoring.

Questions/answers stay in the evaluation annotation, never in the compression
manifest. No OCR calls are made by preparation, configuration, or unit tests.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys
from .utils import atomic_json, canonical_hash, sha256_file, jsonl_read

OFFICIAL_REPO = 'https://github.com/Yuliang-Liu/MultimodalOCR.git'
OFFICIAL_COMMIT = '0ea56c2503a2d940700d581b777f23c16de0eafd'
DEFAULT_VLM = 'Qwen/Qwen2.5-VL-3B-Instruct'


def annotations(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('OCRBench annotation must be a nonempty JSON array')
    seen = set()
    for row in rows:
        for key in ('id','image_path','question','answers','type','dataset_name'):
            if key not in row: raise ValueError('Missing OCRBench field: '+key)
        ident = canonical_hash([row['dataset_name'],row['type'],row['id']])
        if ident in seen: raise ValueError('Duplicate OCRBench QA identity')
        if not isinstance(row['question'],str) or not isinstance(row['image_path'],str):
            raise ValueError('Invalid question/image path')
        seen.add(ident)
    return rows


def prepare(annotation, images_root, output):
    from .data import record_image
    root, out = Path(images_root).resolve(), Path(output).resolve()
    rows = annotations(annotation); by_path = {}; by_pixel = {}; questions = []
    out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        p = (root / row['image_path']).resolve()
        if not p.is_relative_to(root): raise ValueError('Image path escapes dataset root')
        if not p.is_file(): raise FileNotFoundError(p)
        if str(p) not in by_path:
            record = record_image(p,root,'OCRBenchV2',prompt='',origin='missing',kind='natural',source_split='test')
            by_pixel.setdefault(record['pixel_sha256'],record)
            by_path[str(p)] = by_pixel[record['pixel_sha256']]
        item=copy.deepcopy(row)
        item.pop('predict',None);item.pop('score',None)
        item['_aigi_image_id']=by_path[str(p)]['id']
        item['_aigi_original_sha256']=by_path[str(p)]['sha256']
        item['_aigi_original_path']=by_path[str(p)]['image']
        questions.append(item)
    manifest=out/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in by_pixel.values()),encoding='utf-8')
    atomic_json(questions,out/'questions.json')
    atomic_json({'annotation_sha256':sha256_file(annotation),'questions':len(questions),'unique_images':len(by_pixel),
                 'manifest_sha256':sha256_file(manifest),'questions_sha256':sha256_file(out/'questions.json'),
                 'compression_prompt':'empty or independently generated image-only caption; NEVER QA text',
                 'official_commit':OFFICIAL_COMMIT},out/'prepare.json')
    return str(manifest)


def resolve_inputs(questions, inference=None):
    mapping=None
    if inference:
        report=json.loads(Path(inference).read_text())
        mapping={r['id']:r for r in report['samples']}
        if len(mapping)!=len(report['samples']):raise ValueError('Duplicate reconstruction ID')
        if set(mapping)!={q['_aigi_image_id'] for q in questions}:
            raise ValueError('Reconstructed image set must exactly match the QA image set')
    resolved=[]
    for q in questions:
        path=Path(q['_aigi_original_path'])
        if mapping:
            r=mapping[q['_aigi_image_id']]
            if r['image_sha256']!=q['_aigi_original_sha256']:
                raise ValueError('Reconstruction belongs to different reference content')
            path=Path(r['reconstruction'])
        if not path.is_file():raise FileNotFoundError(path)
        actual=sha256_file(path)
        if mapping is None and actual!=q['_aigi_original_sha256']:
            raise ValueError('Original image changed')
        resolved.append((str(path.resolve()),actual))
    return resolved


def model_fingerprint(root):
    root=Path(root)
    if not root.is_dir():raise FileNotFoundError('Download the VLM explicitly before prediction: '+str(root))
    paths=sorted(p for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts)
    if not paths:raise ValueError('Empty VLM directory')
    return canonical_hash({str(p.relative_to(root)):sha256_file(p) for p in paths})


def predict(questions_path, model_path, output, inference=None, device='cuda', max_tokens=1024,
            min_pixels=256*28*28, max_pixels=1280*28*28, seed=903):
    if max_tokens<1 or min_pixels<28*28 or max_pixels<min_pixels:
        raise ValueError('Invalid generation/vision token budget')
    rows=annotations(questions_path);inputs=resolve_inputs(rows,inference)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    cache=output.with_suffix(output.suffix+'.progress.jsonl');lock=output.with_suffix(output.suffix+'.contract.json')
    from .utils import runtime_fingerprint, seed_all, preload_hf_datasets
    contract={'questions_sha256':sha256_file(questions_path),'images_sha256':canonical_hash(inputs),
              'vlm_sha256':model_fingerprint(model_path),'max_new_tokens':max_tokens,
              'min_pixels':min_pixels,'max_pixels':max_pixels,'seed':seed,'device':str(device),
              'runtime':runtime_fingerprint(),'evaluator':'Qwen2.5-VL-local-v1',
              'source':'reconstruction' if inference else 'original','inference_sha256':sha256_file(inference) if inference else None}
    if lock.exists():
        if json.loads(lock.read_text())!=contract:raise ValueError('OCR resume contract changed; use a new output')
    else:
        if cache.exists() or output.exists():raise ValueError('Existing predictions have no contract')
        atomic_json(contract,lock)
    done={}
    if cache.exists():
        for item in jsonl_read(cache):
            i=item['index']
            if not isinstance(i,int) or not 0<=i<len(rows) or i in done or not isinstance(item['predict'],str):
                raise ValueError('Corrupt/duplicate OCR resume row')
            done[i]=item['predict']
    if len(done)<len(rows):
        preload_hf_datasets()
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from .data import open_rgb
        seed_all(seed,True)
        dtype=torch.bfloat16 if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
        processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=min_pixels,max_pixels=max_pixels)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=dtype,
                                                                attn_implementation='sdpa').to(device).eval()
        for i,(row,(path,_)) in enumerate(zip(rows,inputs)):
            if i in done:continue
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=processor(text=[text],images=[open_rgb(path)],padding=True,return_tensors='pt').to(device)
            with torch.inference_mode():
                generated=model.generate(**batch,do_sample=False,max_new_tokens=max_tokens)
            answer=processor.batch_decode(generated[:,batch.input_ids.shape[1]:],skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
            with cache.open('a',encoding='utf-8') as f:
                f.write(json.dumps({'index':i,'predict':answer},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            done[i]=answer
    predictions=[]
    for i,row in enumerate(rows):
        item={k:v for k,v in row.items() if not k.startswith('_aigi_')}
        item['predict']=done[i];predictions.append(item)
    atomic_json(predictions,output)
    return str(output)


def verify_official(root):
    root=Path(root).resolve()
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if commit!=OFFICIAL_COMMIT:raise ValueError('OCRBench evaluator revision mismatch')
    subprocess.run(['git','-C',str(root),'diff','--exit-code',OFFICIAL_COMMIT,'--','OCRBench_v2/eval_scripts'],check=True)
    return root/'OCRBench_v2'


def score(predictions, official_root, output, python_executable=None, execute=False):
    rows=annotations(predictions)
    if any(not isinstance(r.get('predict'),str) for r in rows):raise ValueError('All QA rows need predictions before scoring')
    root=Path(official_root).resolve()/'OCRBench_v2';out=Path(output).resolve()
    py=python_executable or sys.executable
    commands=[[py,str(root/'eval_scripts/eval.py'),'--input_path',str(Path(predictions).resolve()),'--output_path',str(out/'scored.json')],
              [py,str(root/'eval_scripts/get_score.py'),'--json_file',str(out/'scored.json')]]
    plan={'commands':commands,'cwd':str(root),'official_commit':OFFICIAL_COMMIT,'execute':execute}
    if not execute:print(json.dumps(plan,indent=2));return plan
    root=verify_official(official_root);out.mkdir(parents=True,exist_ok=True)
    for i,argv in enumerate(commands):
        with (out/f'official_{i}.log').open('w') as log:
            subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    scored=json.loads((out/'scored.json').read_text())
    if len(scored)!=len(rows):raise ValueError('Official evaluator dropped samples')
    identity=lambda r:canonical_hash([r['dataset_name'],r['type'],r['id']])
    if {identity(r) for r in scored}!={identity(r) for r in rows}:raise ValueError('Official scored QA identities changed')
    import math
    if any(not isinstance(r.get('score'),(int,float)) or not math.isfinite(r['score']) for r in scored):
        raise ValueError('Official scorer did not produce finite scores for every QA item')
    # Preserve the official aggregator's stdout verbatim rather than inventing
    # a replacement aggregate across differently weighted benchmark categories.
    atomic_json({**plan,'predictions_sha256':sha256_file(predictions),'questions':len(scored),
                 'scored_sha256':sha256_file(out/'scored.json'),
                 'official_summary':(out/'official_1.log').read_text()},out/'report.json')
    return str(out/'report.json')


def assets(output, download=False):
    out=Path(output).resolve()
    plan={'vlm':DEFAULT_VLM,'official_repo':OFFICIAL_REPO,'official_commit':OFFICIAL_COMMIT,'output':str(out)}
    if not download:print(json.dumps(plan,indent=2));return plan
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(DEFAULT_VLM)
    out.mkdir(parents=True,exist_ok=True)
    snapshot_download(DEFAULT_VLM,revision=info.sha,local_dir=out/'vlm')
    target=out/'MultimodalOCR'
    if not target.exists():subprocess.run(['git','clone','--no-checkout',OFFICIAL_REPO,str(target)],check=True)
    else:
        subprocess.run(['git','-C',str(target),'diff','--exit-code'],check=True)
        subprocess.run(['git','-C',str(target),'diff','--cached','--exit-code'],check=True)
    subprocess.run(['git','-C',str(target),'fetch','origin',OFFICIAL_COMMIT],check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',OFFICIAL_COMMIT],check=True)
    verify_official(target)
    atomic_json({**plan,'vlm_revision':info.sha,'vlm_sha256':model_fingerprint(out/'vlm')},out/'assets.lock.json')
    return str(out/'assets.lock.json')


def export_hf(output, local_source=None, revision='c7e7cdf23bdb6774661e9b0caf0d9935a42feb8b', limit=0):
    import tempfile
    if limit<0:raise ValueError('limit must be nonnegative')
    out=Path(output).resolve()
    request={'repo':'ling99/OCRBench_v2','revision':revision,'split':'test','image':'image',
             'prompt':None,'kind':'natural','ocrbench':True,'output':str(out/'export'),
             'local_source':str(Path(local_source).resolve()) if local_source else None,'limit':limit}
    with tempfile.TemporaryDirectory(prefix='aigi_ocr_export_') as td:
        path=Path(td)/'request.json';path.write_text(json.dumps(request))
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('hf_worker.py')),'--request',str(path)],
                       cwd=td,env=env,check=True)
    prepared=prepare(out/'export/OCRBench_v2.json',out/'export',out/'prepared')
    return prepared

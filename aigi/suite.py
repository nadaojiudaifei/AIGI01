"""Supplementary executable experiment plans, never fabricated measurements."""
from pathlib import Path
import json
import sys
from .utils import atomic_json


def build_suite(matrix, output, questions=None, vlm=None, official=None, manifest=None, eval_python=None):
    source=json.loads(Path(matrix).read_text());root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    ocr_requested=any(x is not None for x in (questions,vlm,official,manifest))
    if ocr_requested and not all((questions,vlm,official,manifest)):
        raise ValueError('OCR suite requires questions, model, official-root and manifest together')
    finals={}
    for job in source['jobs']:
        if job.get('kind')=='train':
            # One terminal stage per variant and quality.
            key=job['id'].rsplit('_',1)[0];finals[key]=job
    jobs=[]
    def add(ident,argv,expected,dependency,config=None):
        job={'id':ident,'argv':argv,'expected':str(expected),'dependency':str(dependency) if dependency else None}
        if config:job['config']=str(config)
        jobs.append(job)
    def score_job(label,prediction):
        out=root/label/'official';argv=[sys.executable,'-m','aigi','ocr-score','--predictions',str(prediction),
                                      '--official-root',str(Path(official).resolve()),'--output',str(out),'--execute']
        if eval_python:argv+=['--eval-python',eval_python]
        add(label+'_score',argv,out/'report.json',prediction)
    if ocr_requested:
        original=root/'original/predictions.json'
        add('original_ocr',[sys.executable,'-m','aigi','ocr-predict','--questions',str(Path(questions).resolve()),
                            '--model',str(Path(vlm).resolve()),'--output',str(original)],original,questions)
        score_job('original',original)
    for label,job in finals.items():
        cfg=job['config'];checkpoint=job['expected'];work=root/label
        add(label+'_profile',[sys.executable,'-m','aigi','profile','--config',cfg,'--checkpoint',checkpoint,
                               '--output',str(work/'profile.json')],work/'profile.json',checkpoint,cfg)
        if ocr_requested:
            inference=work/'ocr_images/inference.json';prediction=work/'predictions.json'
            add(label+'_ocr_images',[sys.executable,'-m','aigi','infer','--config',cfg,'--checkpoint',checkpoint,
                                    '--manifest',str(Path(manifest).resolve()),'--output',str(inference.parent)],inference,checkpoint,cfg)
            add(label+'_ocr_predict',[sys.executable,'-m','aigi','ocr-predict','--questions',str(Path(questions).resolve()),
                                      '--model',str(Path(vlm).resolve()),'--inference',str(inference),'--output',str(prediction)],prediction,inference)
            score_job(label,prediction)
    if not jobs:raise ValueError('Source matrix has no training jobs')
    atomic_json({'version':1,'jobs':jobs,'source_matrix':str(Path(matrix).resolve()),
                 'note':'Run only after base matrix. Human responses are never generated automatically; build study separately after rate matching.'},root/'matrix.json')
    return str(root/'matrix.json')

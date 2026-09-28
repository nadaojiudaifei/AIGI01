"""Blinded paired-study materials and participant-cluster inference.

No scores are produced without real response files. Participant codes are
anonymous; the browser exports CSV locally and performs no network requests.
"""
from pathlib import Path
import csv
import json
import random
import shutil
import itertools
import math
from PIL import Image
import numpy as np
from .utils import atomic_json, canonical_hash, sha256_file


def build_study(first, second, output, seed=903, tolerance=.05, criterion='realism'):
    if criterion not in {'realism', 'fidelity'} or not 0 <= tolerance < 1:
        raise ValueError('Invalid criterion/rate tolerance')
    inputs = [json.loads(Path(p).read_text()) for p in (first,second)]
    labels = [d.get('experiment',d['method']) for d in inputs]
    if labels[0] == labels[1] or 'tie' in labels:
        raise ValueError('Study methods/experiment labels must be distinct')
    samples = [{r['id']:r for r in d['samples']} for d in inputs]
    if any(len(s) != len(d['samples']) for s,d in zip(samples,inputs)):
        raise ValueError('Duplicate sample ID')
    if not samples[0] or set(samples[0]) != set(samples[1]):
        raise ValueError('Study reference sets must be nonempty and identical')
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use an empty output directory to prevent stale study trials')
    public = root/'public'; public.mkdir(parents=True)
    study_id = canonical_hash({'inputs':[sha256_file(first),sha256_file(second)],
                               'seed':seed,'criterion':criterion,'tolerance':tolerance})[:24]
    rng, trials = random.Random(seed), []
    ids = sorted(samples[0]); rng.shuffle(ids)
    for index, ident in enumerate(ids):
        rows = [s[ident] for s in samples]
        if rows[0]['image_sha256'] != rows[1]['image_sha256']:
            raise ValueError('Reference content mismatch')
        rates = [r['bpp_total'] for r in rows]
        if not all(isinstance(r,(int,float)) and math.isfinite(r) and r>0 for r in rates) or abs(rates[0]-rates[1])/max(rates) > tolerance:
            raise ValueError('Rates do not satisfy the declared matching tolerance')
        order = [0,1]; rng.shuffle(order)
        paths, key = [], {}
        for side, chosen in zip(('A','B'),order):
            path = f'trial_{index:04d}_{side}.png'
            with Image.open(rows[chosen]['reconstruction']) as image:
                image.convert('RGB').save(public/path)
            paths.append(path); key[side] = labels[chosen]
        ref = None
        if criterion == 'fidelity':
            ref = f'trial_{index:04d}_reference.png'
            if sha256_file(rows[0]['original']) != rows[0]['image_sha256']:raise ValueError('Reference file changed')
            with Image.open(rows[0]['original']) as image:
                image.convert('RGB').save(public/ref)
        trials.append({'id':str(index),'image_id':ident,'images':paths,'reference':ref,
                       'key':key,'bpp':dict(zip(labels,rates)),
                       'display_sha256':{p:sha256_file(public/p) for p in paths}})
    sections=[]
    for t in trials:
        reference = f'<p>Reference</p><img class="ref" src="{t["reference"]}">' if t['reference'] else ''
        sections.append(f'<section><h2>Trial {t["id"]}</h2>{reference}<div>'+
                        ''.join(f'<figure><img src="{p}"><figcaption>{s}</figcaption></figure>' for p,s in zip(t['images'],('A','B')))+
                        f'</div><label>Choice <select data-id="{t["id"]}"><option value="">Unanswered</option><option>A</option><option>B</option><option>Tie</option></select></label></section>')
    instruction = ('Choose the visually more realistic reconstruction. No original is displayed.' if criterion=='realism'
                   else 'Choose the reconstruction closer to the reference while retaining natural appearance.')
    page='''<!doctype html><meta charset="utf-8"><title>Blind image comparison</title>
<style>body{font-family:sans-serif;max-width:1500px;margin:24px auto}section{border-top:1px solid;padding:20px}figure{display:inline-block;width:46%;vertical-align:top;margin:1%}figure img{width:100%}.ref{max-width:95%}</style>
<h1>Blind comparison</h1><p>Participation is voluntary. Use an anonymous code, not your name/email. Responses are saved on your device only. Keep browser zoom and viewing conditions unchanged.</p>'''
    page+=f'<p>{instruction}</p><label>Anonymous code <input id="participant"></label><button onclick="save()">Export answers</button>'+'\n'.join(sections)
    page+='<script>const studyId='+json.dumps(study_id)+';const criterion='+json.dumps(criterion)+';'+'''
function save(){const p=document.querySelector('#participant').value;
if(!/^[a-zA-Z0-9_-]{1,40}$/.test(p)){alert('Use an anonymous alphanumeric code');return;}
const selects=[...document.querySelectorAll('select')];if(selects.some(s=>!s.value)){alert('Answer every trial before exporting');return;}
let rows=['participant,study_id,trial,choice,criterion'];selects.forEach(s=>rows.push([p,studyId,s.dataset.id,s.value,criterion].join(',')));
const url=URL.createObjectURL(new Blob([rows.join('\\n')],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='answers_'+p+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
</script>'''
    (public/'index.html').write_text(page,encoding='utf-8')
    atomic_json({'study_id':study_id,'seed':seed,'methods':labels,'criterion':criterion,'trials':trials,
                 'rate_tolerance':tolerance,'version':2,'note':'Distribute only public/. Keep this key private.'},root/'PRIVATE_KEY.json')
    return str(public/'index.html')


def cluster_statistics(matrix, seed=903, draws=10000):
    """Complete participant x image matrix; ties score 0.5. No iid-vote CI."""
    matrix=np.asarray(matrix,dtype=float)
    if matrix.ndim!=2 or min(matrix.shape)<1 or draws<100:
        raise ValueError('Need a nonempty participant-by-image matrix and >=100 bootstrap draws')
    if not np.isin(matrix,[0.,.5,1.]).all():
        raise ValueError('Invalid preference scores')
    participant=matrix.mean(1); images=matrix.mean(0); rng=np.random.default_rng(seed)
    def boot(means):
        if len(means)<2:return None
        values=[float(rng.choice(means,len(means),replace=True).mean()) for _ in range(draws)]
        return [float(v) for v in np.quantile(values,[.025,.975])]
    pci, ici = boot(participant),boot(images)
    centered=participant-.5; observed=abs(centered.mean())
    if len(participant)<2:
        pvalue=None; kind='insufficient_participants'
    elif len(participant)<=16:
        count=sum(abs(np.dot(signs,centered)/len(centered))>=observed-1e-12 for signs in itertools.product((-1,1),repeat=len(centered)))
        pvalue=count/(2**len(centered));kind='exact_participant_sign_flip'
    else:
        count=sum(abs(np.dot(rng.choice([-1,1],len(centered)),centered)/len(centered))>=observed-1e-12 for _ in range(draws))
        pvalue=(count+1)/(draws+1);kind='monte_carlo_participant_sign_flip'
    return {'preference_including_half_ties':float(matrix.mean()),'participant_cluster_ci95':pci,
            'image_cluster_ci95_sensitivity':ici,'participant_sign_flip_p_two_sided':pvalue,
            'test':kind,'draws':draws,'seed':seed,
            'assumptions':'Complete crossed panel. Primary interval conditions on these images and resamples independent participants. Image interval is a separate sensitivity analysis, not a two-way joint CI.'}


def aggregate_study(key_path, answers, output, seed=903, draws=10000):
    key=json.loads(Path(key_path).read_text()); labels=key['methods']
    trials={t['id']:t for t in key['trials']}
    if len(labels)!=2 or labels[0]==labels[1] or not trials:
        raise ValueError('Not a two-method paired study')
    votes={}; counts={labels[0]:0,labels[1]:0,'tie':0}
    for path in answers:
        with Path(path).open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('study_id')!=key['study_id'] or row.get('criterion')!=key['criterion']:
                    raise ValueError('Study ID/criterion mismatch')
                participant=row.get('participant',''); trial=row.get('trial',''); choice=row.get('choice','')
                if not participant or trial not in trials or choice not in {'A','B','Tie'}:
                    raise ValueError('Malformed or unanswered response')
                identity=(participant,trial)
                if identity in votes: raise ValueError('Duplicate participant/trial response')
                winner='tie' if choice=='Tie' else trials[trial]['key'][choice]
                counts[winner]+=1
                votes[identity]=.5 if winner=='tie' else float(winner==labels[0])
    participants=sorted({p for p,_ in votes})
    if not participants: raise ValueError('No participant responses; cannot manufacture study results')
    if len(votes)!=len(participants)*len(trials):
        raise ValueError('Incomplete participant panel. No silent exclusions are allowed.')
    matrix=[[votes[(p,t)] for t in trials] for p in participants]
    result={'study_id':key['study_id'],'criterion':key['criterion'],'methods':labels,
            'participants':len(participants),'images':len(trials),'responses':len(votes),'counts':counts,
            'first_method_statistics':cluster_statistics(matrix,seed,draws),
            'source_sha256':{str(Path(p).resolve()):sha256_file(p) for p in answers},
            'note':'This analyzes supplied responses only; it does not assert participant recruitment, consent, representativeness or equivalence to the original paper.'}
    atomic_json(result,output);return result

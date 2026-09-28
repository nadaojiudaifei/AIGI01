"""Dependency-light CLI. Heavy neural libraries load only for their own command."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def parser():
    p = argparse.ArgumentParser(prog='python -m aigi', description='AIGI01: additive DiT-IC research implementation')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('catalog', help='Print verified dataset/model catalog; no download')
    q = sub.add_parser('models', help='Plan model download; --download explicitly executes it')
    q.add_argument('--output', default='aigi_assets'); q.add_argument('--qualities', nargs='+', type=int, choices=range(1,5), default=[3])
    q.add_argument('--download', action='store_true'); q.add_argument('--no-caption', action='store_true')
    q = sub.add_parser('model-url', help='Download an HTTPS/GitHub-release weight with required SHA256')
    for k in ('url','output','sha256'): q.add_argument('--'+k, required=True)
    q = sub.add_parser('data-hf', help='Export a catalog HF dataset or its already-downloaded local copy')
    q.add_argument('name'); q.add_argument('--output', required=True); q.add_argument('--split'); q.add_argument('--limit', type=int, default=0)
    q.add_argument('--local-source'); q.add_argument('--revision'); q.add_argument('--hf-config'); q.add_argument('--first', type=int); q.add_argument('--last', type=int)
    q.add_argument('--allow-large', action='store_true')
    q = sub.add_parser('data-local', help='Build manifest from local image folder and prompt sidecars')
    q.add_argument('--root', required=True); q.add_argument('--output', required=True); q.add_argument('--dataset', default='local')
    q.add_argument('--kind', choices=['natural','aigi'], default='natural'); q.add_argument('--split', choices=['train','validation','test'], default='train'); q.add_argument('--expected', type=int)
    q = sub.add_parser('split', help='Deduplicate pixels; group original prompts; fixed 98/1/1 split')
    q.add_argument('--inputs', nargs='+', required=True); q.add_argument('--output', required=True); q.add_argument('--seed',type=int,default=903)
    q = sub.add_parser('caption', help='Generate BLIP captions ONLY for missing prompts')
    q.add_argument('--manifest', required=True); q.add_argument('--output', required=True); q.add_argument('--model',default='aigi_assets/caption'); q.add_argument('--device',default='cuda')
    q = sub.add_parser('configure', help='Resolve asset paths and write a strict new configuration')
    q.add_argument('--base'); q.add_argument('--output', required=True); q.add_argument('--assets'); q.add_argument('--method'); q.add_argument('--quality',type=int)
    q.add_argument('--set',action='append',default=[],metavar='KEY=JSON_VALUE')
    for name in ('teacher','train','infer','encode','decode','profile','acceptance'):
        q = sub.add_parser(name)
        q.add_argument('--config', required=True); q.add_argument('--device',default='cuda')
        if name in ('infer','encode','decode','profile'): q.add_argument('--checkpoint'); q.add_argument('--ema',action='store_true')
        if name in ('teacher','infer'): q.add_argument('--manifest',required=True); q.add_argument('--limit',type=int,default=0)
        if name in ('infer','encode','decode','profile'): q.add_argument('--output',required=True)
        if name in ('encode','decode'): q.add_argument('--input',required=True); q.add_argument('--prompt-file')
        if name=='encode': q.add_argument('--external-prompt',action='store_true')
        if name=='train': q.add_argument('--resume'); q.add_argument('--stop-after',type=int)
        if name=='profile': q.add_argument('--sizes',nargs='+',type=int,default=[1024,2048,4096]); q.add_argument('--repeats',type=int,default=5)
        if name=='acceptance': q.add_argument('--output',required=True); q.add_argument('--execute',action='store_true'); q.add_argument('--extended',action='store_true'); q.add_argument('--gpus',type=int,default=1)
    q = sub.add_parser('evaluate',help='Compute native-resolution metrics from an inference.json')
    q.add_argument('--input',required=True); q.add_argument('--output',required=True); q.add_argument('--device',default='cuda')
    q.add_argument('--metrics',nargs='+',default=['psnr','ms_ssim','lpips','dists','niqe','clipiqa','musiq','fid','kid']); q.add_argument('--allow-download',action='store_true')
    q = sub.add_parser('warmup', help='Explicitly download/cache perceptual metric models on the remote server')
    q.add_argument('--device',default='cuda')
    q = sub.add_parser('matrix',help='Generate experiment plan; NEVER starts training')
    q.add_argument('--config',required=True); q.add_argument('--output',required=True); q.add_argument('--qualities',nargs='+',type=int,choices=range(1,5),default=[1,2,3,4])
    q.add_argument('--protocol',choices=['main','ablation256'],default='main')
    q.add_argument('--no-ablations',action='store_true'); q.add_argument('--no-add-one',action='store_true'); q.add_argument('--factorial',choices=['idea1','idea2'])
    q.add_argument('--max-jobs',type=int,default=1000); q.add_argument('--gpus',type=int,default=1); q.add_argument('--datasets',nargs='*',default=[])
    q = sub.add_parser('run',help='Print experiment plan, or run sequentially with --execute')
    q.add_argument('--plan',required=True); q.add_argument('--execute',action='store_true')
    q = sub.add_parser('compare'); q.add_argument('--inputs',nargs='+',required=True); q.add_argument('--output',required=True)
    q.add_argument('--metric',default='psnr'); q.add_argument('--baseline',default='ditic'); q.add_argument('--plot',action='store_true')
    q = sub.add_parser('study',help='Generate blinded human-study material; no fabricated scores')
    q.add_argument('--a',required=True); q.add_argument('--b',required=True); q.add_argument('--output',required=True); q.add_argument('--seed',type=int,default=903)
    q.add_argument('--criterion',choices=['realism','fidelity'],default='realism');q.add_argument('--tolerance',type=float,default=.05)
    q = sub.add_parser('study-results',help='Analyze supplied anonymous complete-panel study responses')
    q.add_argument('--key',required=True);q.add_argument('--answers',nargs='+',required=True);q.add_argument('--output',required=True)
    q.add_argument('--seed',type=int,default=903);q.add_argument('--draws',type=int,default=10000)
    q = sub.add_parser('ocr-assets',help='Plan OCR evaluator assets; --download explicitly fetches them')
    q.add_argument('--output',required=True);q.add_argument('--download',action='store_true')
    q = sub.add_parser('ocr-data',help='Export the pinned OCRBench V2 HF dataset; never calls OCR')
    q.add_argument('--output',required=True);q.add_argument('--local-source');q.add_argument('--limit',type=int,default=0)
    q = sub.add_parser('ocr-prepare',help='Prepare local official QA and a unique-image compression manifest')
    q.add_argument('--annotations',required=True);q.add_argument('--images-root',required=True);q.add_argument('--output',required=True)
    q = sub.add_parser('ocr-predict',help='Run a fixed local Qwen2.5-VL on original or reconstructed images')
    q.add_argument('--questions',required=True);q.add_argument('--model',required=True);q.add_argument('--output',required=True)
    q.add_argument('--inference');q.add_argument('--device',default='cuda');q.add_argument('--max-tokens',type=int,default=1024)
    q.add_argument('--min-pixels',type=int,default=200704);q.add_argument('--max-pixels',type=int,default=1003520)
    q = sub.add_parser('ocr-score',help='Plan/run the pinned original OCRBench V2 scoring scripts')
    q.add_argument('--predictions',required=True);q.add_argument('--official-root',required=True);q.add_argument('--output',required=True)
    q.add_argument('--eval-python');q.add_argument('--execute',action='store_true')
    q = sub.add_parser('checkpoint-compare',help='Exact tensor/state comparison of uninterrupted vs resumed training')
    q.add_argument('--a',required=True);q.add_argument('--b',required=True);q.add_argument('--output',required=True)
    q = sub.add_parser('suite',help='Create supplementary OCR, profiling and human-study jobs for a completed matrix')
    q.add_argument('--matrix',required=True);q.add_argument('--output',required=True);q.add_argument('--ocr-questions')
    q.add_argument('--ocr-model');q.add_argument('--ocr-official');q.add_argument('--ocr-manifest');q.add_argument('--eval-python')
    q = sub.add_parser('inspect',help='Inspect a packet without a model or original image')
    q.add_argument('--input',required=True); q.add_argument('--prompt-file')
    q = sub.add_parser('doctor',help='Read-only dependency/asset diagnostics; installs nothing')
    q.add_argument('--config'); q.add_argument('--output',default='aigi_runs/doctor.json'); q.add_argument('--require-gpu',action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cmd = args.command
    if cmd == 'catalog':
        from .registry import DATASETS, MODELS
        print(json.dumps(dict(datasets=DATASETS,models=MODELS),indent=2,ensure_ascii=False)); return 0
    if cmd == 'models':
        from .assets import download_models
        download_models(args.output,args.qualities,not args.no_caption,plan=not args.download); return 0
    if cmd == 'model-url':
        from .assets import download_url
        download_url(args.url,args.output,args.sha256); return 0
    if cmd == 'data-hf':
        from .assets import export_hf, diffusiondb, mlic_download
        from .registry import DATASETS
        if args.name not in DATASETS: raise ValueError('Unknown dataset; run catalog')
        spec=DATASETS[args.name]
        if args.limit < 0: raise ValueError('limit must be nonnegative')
        if spec['loader']=='diffusiondb':
            if args.limit: raise ValueError('Use --first/--last for archive subsets, not --limit')
            count=(args.last or spec['last'])-(args.first or spec['first'])+1
            if count>10 and not args.allow_large: raise ValueError('More than 10 archive parts requires --allow-large')
            diffusiondb(args.name,args.output,args.first,args.last,args.local_source,args.revision)
        elif spec['loader']=='7z':
            if not args.allow_large: raise ValueError('MLIC-Train-100K full archive requires --allow-large')
            mlic_download(args.output,args.local_source,args.revision)
        else: export_hf(args.name,args.output,args.split,args.limit,args.local_source,args.revision,args.hf_config)
        return 0
    if cmd=='data-local':
        from .data import build_local_manifest
        print(build_local_manifest(args.root,args.output,args.dataset,args.kind,args.split,args.expected)); return 0
    if cmd=='split':
        from .data import split_manifests
        print(split_manifests(args.inputs,args.output,args.seed)); return 0
    if cmd=='caption':
        from .caption import caption_manifest
        caption_manifest(args.manifest,args.output,args.model,args.device); return 0
    if cmd=='configure':
        import yaml
        from .config import Config, load_config, save_config
        c=load_config(args.base) if args.base else Config()
        if args.method: c.method=args.method
        if args.quality: c.quality=args.quality
        if args.assets:
            from .assets import configure
            c=configure(c,args.assets)
        for item in args.set:
            key,value=item.split('=',1)
            if key.startswith('components.'):
                obj,key=c.components,key.split('.',1)[1]
            else: obj=c
            if not hasattr(obj,key): raise ValueError('Unknown configuration key: '+key)
            setattr(obj,key,yaml.safe_load(value))
        save_config(c.validate(),args.output); print(args.output); return 0
    if cmd=='doctor':
        from .doctor import doctor
        return doctor(args.config,args.output,args.require_gpu)
    if cmd=='warmup':
        from .utils import preload_hf_datasets
        preload_hf_datasets()
        import torch, lpips, pyiqa
        from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
        lpips.LPIPS(net='alex').to(args.device)
        for name in ('dists','ms_ssim','niqe','clipiqa','musiq'):
            metric=pyiqa.create_metric(name,device=args.device); del metric
        FrechetInceptionDistance().to(args.device); KernelInceptionDistance(subset_size=1000).to(args.device)
        print('Metric assets instantiated. Future offline runs must use the same cache locations.'); return 0
    if cmd=='evaluate':
        from .evaluate import evaluate
        evaluate(args.input,args.output,args.metrics,args.device,offline=not args.allow_download); return 0
    if cmd=='compare':
        from .compare import compare
        compare(args.inputs,args.output,args.metric,args.baseline,args.plot); return 0
    if cmd=='run':
        from .experiments import run_matrix
        run_matrix(args.plan,args.execute); return 0
    if cmd=='study':
        from .study import build_study
        build_study(args.a,args.b,args.output,args.seed,args.tolerance,args.criterion); return 0
    if cmd=='study-results':
        from .study import aggregate_study
        aggregate_study(args.key,args.answers,args.output,args.seed,args.draws);return 0
    if cmd.startswith('ocr-'):
        from . import ocrbench
        if cmd=='ocr-assets':ocrbench.assets(args.output,args.download)
        elif cmd=='ocr-data':ocrbench.export_hf(args.output,args.local_source,limit=args.limit)
        elif cmd=='ocr-prepare':ocrbench.prepare(args.annotations,args.images_root,args.output)
        elif cmd=='ocr-predict':ocrbench.predict(args.questions,args.model,args.output,args.inference,args.device,args.max_tokens,args.min_pixels,args.max_pixels)
        elif cmd=='ocr-score':ocrbench.score(args.predictions,args.official_root,args.output,args.eval_python,args.execute)
        return 0
    if cmd=='checkpoint-compare':
        from .checkpoint import compare_training
        compare_training(args.a,args.b,args.output);return 0
    if cmd=='suite':
        from .suite import build_suite
        build_suite(args.matrix,args.output,args.ocr_questions,args.ocr_model,args.ocr_official,args.ocr_manifest,args.eval_python);return 0
    if cmd=='inspect':
        from .bitstream import unpack, rate_report, MAX_PAYLOAD, MAX_HEADER
        p=Path(args.input)
        if p.stat().st_size>MAX_PAYLOAD+MAX_HEADER+44: raise ValueError('Oversized stream')
        raw=p.read_bytes(); prompt=Path(args.prompt_file).read_text() if args.prompt_file else None
        packet=unpack(raw,prompt)
        print(json.dumps(dict(metadata=packet.metadata,prompt=packet.prompt,rates=rate_report(raw)),indent=2,ensure_ascii=False)); return 0
    from .config import load_config
    cfg=load_config(args.config)
    if cmd=='teacher':
        from .teacher import build_cache
        build_cache(cfg,args.manifest,args.device,args.limit)
    elif cmd=='train':
        from .train import train
        train(cfg,args.resume,args.device,args.stop_after)
    elif cmd=='matrix':
        from .experiments import build_matrix
        print(build_matrix(cfg,args.output,args.qualities,not args.no_ablations,not args.no_add_one,args.factorial,args.max_jobs,args.gpus,args.datasets,args.protocol))
    elif cmd=='acceptance':
        from .acceptance import acceptance
        acceptance(cfg,args.output,args.execute,args.device,args.extended,args.gpus)
    else:
        from .infer import CodecSession, infer_manifest
        session=CodecSession(cfg,args.checkpoint,args.device,args.ema)
        if cmd=='infer': infer_manifest(session,args.manifest,args.output,args.limit)
        elif cmd=='encode':
            prompt=Path(args.prompt_file).read_text(encoding='utf-8') if args.prompt_file else ''
            print(json.dumps(session.encode_file(args.input,args.output,prompt,args.external_prompt)))
        elif cmd=='decode':
            prompt=Path(args.prompt_file).read_text(encoding='utf-8') if args.prompt_file else None
            print(json.dumps(session.decode_file(args.input,args.output,prompt)))
        elif cmd=='profile':
            from .profile import profile
            profile(session,args.output,args.sizes,repeats=args.repeats)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f'AIGI01 error: {exc}',file=sys.stderr)
        raise SystemExit(2)

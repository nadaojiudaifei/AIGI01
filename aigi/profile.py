"""Uninstrumented latency + separate per-module diagnostic FLOPs/timing."""
import time
import statistics
import torch
from .utils import atomic_json
from .instrumentation import instrument, observed_modules


@torch.no_grad()
def profile(session, output, sizes=(1024,2048,4096), warmup=2, repeats=5):
    if not str(session.device).startswith('cuda'):
        raise ValueError('Reported GPU profile requires CUDA')
    if not sizes or repeats<1 or warmup<0:raise ValueError('Invalid profile repetition settings')
    rows=[]; session.sync();start=time.perf_counter()
    t,m,n,nm=session.get_text('a natural photograph');session.sync()
    text_cold_ms=(time.perf_counter()-start)*1000
    for size in sizes:
        if size<256 or size>8192 or size%256:raise ValueError('Profile sizes must be multiples of 256 in [256,8192]')
        x=torch.zeros(1,3,size,size,device=session.device);enc=[];dec=[];nfe=[]
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(warmup+repeats):
            session.sync();begin=time.perf_counter()
            code=session.model.compress(x,t,m)
            session.sync();middle=time.perf_counter()
            image,extra=session.model.decompress(code['strings'],code['shape'],t,m,n,nm)
            session.sync();end=time.perf_counter()
            if iteration>=warmup:
                enc.append((middle-begin)*1000);dec.append((end-middle)*1000);nfe.append(extra['nfe'])
            del image
        peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
        modules=observed_modules(session.model)
        code,encode_detail=instrument(lambda:session.model.compress(x,t,m),modules,session.sync)
        decoded,decode_detail=instrument(lambda:session.model.decompress(code['strings'],code['shape'],t,m,n,nm),modules,session.sync)
        del decoded
        rows.append({'size':size,'encode_ms':statistics.mean(enc),'decode_ms':statistics.mean(dec),
                     'encode_median_ms':statistics.median(enc),'decode_median_ms':statistics.median(dec),
                     'encode_samples_ms':enc,'decode_samples_ms':dec,'nfe':nfe[0],
                     'peak_cuda_allocated_bytes':peak_allocated,'peak_cuda_reserved_bytes':peak_reserved,
                     'encode_modules':encode_detail,'decode_modules':decode_detail})
    result={'method':session.cfg.method,'runtime':session.runtime,'profiles':rows,'text_first_call_ms':text_cold_ms,
            'parameters_total':sum(p.numel() for p in session.model.parameters()),
            'protocol':'FP32;batch1;tiling off;zero-image microbenchmark;CPU entropy incl transfers;separate instrumented pass;not a claim of exact paper-native GPU profile'}
    atomic_json(result,output);return result

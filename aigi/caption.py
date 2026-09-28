"""Caption missing natural/AIGI prompts; never label generated captions as original prompts."""
from pathlib import Path
import json
import torch
from .data import open_rgb
from .utils import jsonl_read


def caption_manifest(source, target, model_path, device='cuda', offline=True):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(model_path, local_files_only=offline)
    model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=offline).to(device).eval().requires_grad_(False)
    target = Path(target)
    if target.resolve() == Path(source).resolve():
        raise ValueError('Write a new manifest; do not overwrite the source')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f, torch.no_grad():
        for row in jsonl_read(source):
            if not row.get('prompt', '').strip():
                inputs = processor(images=open_rgb(row['image']), return_tensors='pt').to(device)
                ids = model.generate(**inputs, do_sample=False, num_beams=3, max_new_tokens=64)
                row['prompt'] = processor.decode(ids[0], skip_special_tokens=True)
                row['prompt_origin'] = 'blip_caption'
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

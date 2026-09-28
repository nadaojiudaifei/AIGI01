"""Frozen, CPU-FP32 Gemma text encoder, identical at both entropy endpoints."""
from pathlib import Path
import hashlib
import torch
from .utils import sha256_file


class TextEncoder:
    def __init__(self, path, offline=True, length=77):
        from transformers import AutoTokenizer, AutoModel
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError('Resolve/download the Sana model to a local path first')
        self.tokenizer = AutoTokenizer.from_pretrained(str(path / 'tokenizer'), local_files_only=offline)
        self.model = AutoModel.from_pretrained(str(path / 'text_encoder'),
                                              local_files_only=offline, torch_dtype=torch.float32,
                                              attn_implementation='eager').cpu().eval().requires_grad_(False)
        self.tokenizer.padding_side = 'right'
        self.length = length
        self._cache = {}
        digest = hashlib.sha256(f'aigi-text-v1;length={length};fp32;eager;no-instruction'.encode())
        files = sorted(p for folder in ('text_encoder', 'tokenizer') for p in (path / folder).rglob('*')
                       if p.is_file() and p.suffix in {'.json', '.model', '.safetensors', '.bin', '.txt'})
        if not files:
            raise FileNotFoundError('No text model files found')
        for p in files:
            digest.update(str(p.relative_to(path)).encode())
            digest.update(bytes.fromhex(sha256_file(p)))
        self.fingerprint = digest.hexdigest()

    @torch.no_grad()
    def __call__(self, prompts, device='cpu'):
        outputs, masks = [], []
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for prompt in prompts:
                if prompt not in self._cache:
                    tokens = self.tokenizer(prompt, padding='max_length', truncation=True,
                                            max_length=self.length, add_special_tokens=True, return_tensors='pt')
                    mask = tokens['attention_mask'].bool()
                    if not bool(mask.any()):
                        raise ValueError('Tokenizer produced no valid tokens')
                    emb = self.model(input_ids=tokens['input_ids'], attention_mask=mask)[0].float()
                    if emb.shape[-1] != 2304:
                        raise ValueError('DiT-IC requires Sana/Gemma caption width 2304')
                    if len(self._cache) > 256:
                        self._cache.clear()
                    self._cache[prompt] = (emb, mask)
                emb, mask = self._cache[prompt]
                outputs.append(emb)
                masks.append(mask)
        finally:
            torch.set_num_threads(old_threads)
        return torch.cat(outputs).to(device), torch.cat(masks).to(device)

    def paired(self, prompts, device='cpu'):
        text, mask = self(prompts, device)
        null, nmask = self([''] * len(prompts), device)
        return text, mask, null, nmask

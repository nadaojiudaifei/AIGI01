"""Reuse unmodified pinned DiT-IC architecture without instantiating its loss object."""
from pathlib import Path
import sys
import torch
from torch import nn


def require_upstream(root):
    root = Path(root).resolve()
    if not (root / 'models/DiT_IC.py').is_file():
        raise FileNotFoundError('DiT-IC not imported. Run aigi_scripts/bootstrap_upstream.py first.')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def tensor_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, dict):
        raise ValueError('Expected a dictionary checkpoint: ' + str(path))
    return data


class Backbone(nn.Module):
    def __init__(self, cfg, device='cuda', training=False):
        super().__init__()
        from .utils import preload_hf_datasets
        preload_hf_datasets()
        root = require_upstream(cfg.upstream_root)
        import yaml
        from diffusers import AutoencoderDC, SanaTransformer2DModel
        from models.latent_codec import LatentCodec
        from models.DiT_IC import LatentConditionAlignment
        from models.scheduler import make_1step_sched, MyDDPMScheduler
        from ELIC.elic_official import ELIC
        params = yaml.safe_load((root / cfg.upstream_config).read_text())['model']['params']
        self.codec_mode = 'self_dist'
        self.DiT_mode = 'scale'
        self.time = int(params.get('time', 999))
        self.register_buffer('timesteps', torch.tensor([self.time], dtype=torch.long))
        merged = cfg.checkpoint_format == 'merged'
        if merged:
            path = str(root / 'SANA')
            self.vae = AutoencoderDC.from_config(AutoencoderDC.load_config(path, subfolder='vae'))
            self.DiT = SanaTransformer2DModel.from_config(SanaTransformer2DModel.load_config(path, subfolder='transformer'))
        else:
            path = cfg.sana_path
            kwargs = dict(local_files_only=cfg.offline, torch_dtype=torch.float32)
            if cfg.model_variant:
                kwargs['variant'] = cfg.model_variant
            self.vae = AutoencoderDC.from_pretrained(path, subfolder='vae', **kwargs)
            self.DiT = SanaTransformer2DModel.from_pretrained(path, subfolder='transformer', **kwargs)
        self.codec = LatentCodec(**params['codec_param'])
        self.prompter = LatentConditionAlignment(320, embed_dim=2304, num_tokens=77)
        self.aux_codec = ELIC().g_a
        scheduler = make_1step_sched(str(root / 'SANA'), self.time, device)
        self.sched = MyDDPMScheduler(scheduler, 'scale', device)
        if not merged:
            elic = ELIC()
            elic.load_state_dict(tensor_checkpoint(cfg.elic_path), strict=True)
            self.aux_codec.load_state_dict(elic.g_a.state_dict(), strict=True)
        self.requires_grad_(False)
        if cfg.checkpoint_format == 'lora':
            self.add_lora(cfg)
            state = tensor_checkpoint(cfg.weights)
            state = state.get('ema', state.get('model', state))
            self.codec.load_state_dict(state['state_dict_codec'], strict=True)
            for module, key in ((self.vae, 'state_dict_vae'), (self.DiT, 'state_dict_DiT'), (self.prompter, 'state_dict_prompter')):
                sd = module.state_dict()
                incoming = state[key]
                if not set(incoming).issubset(sd):
                    raise ValueError('LoRA rank/key mismatch for ' + key)
                sd.update(incoming)
                module.load_state_dict(sd, strict=True)
        elif merged:
            state = tensor_checkpoint(cfg.weights)
            # Loss weights are present in released merged files but not part of codec inference.
            state = {k: v for k, v in state.items() if not k.startswith(('loss.', 'linear_proj.', 'foundation_model.'))}
            self.codec.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith('codec.')}, strict=True)
            self.load_state_dict(state, strict=True)
        if cfg.dit_from_scratch:
            self.DiT = SanaTransformer2DModel.from_config(self.DiT.config)
        if training:
            if cfg.finetune == 'lora' and cfg.checkpoint_format != 'lora':
                self.add_lora(cfg)
            elif cfg.finetune == 'full':
                self.DiT.requires_grad_(True)
                self.vae.decoder.requires_grad_(True)
            self.codec.requires_grad_(True)
            self.prompter.requires_grad_(True)
        else:
            self.requires_grad_(False)
        if hasattr(self.DiT, 'disable_gradient_checkpointing'):
            self.DiT.disable_gradient_checkpointing()  # scoped norm hooks must remain in the first forward graph
        self.aux_codec.requires_grad_(False)
        self.to(device)
        self.codec.update(force=True)

    def add_lora(self, cfg):
        # Invoke upstream helper methods against this identical architectural namespace.
        from models.DiT_IC import Codec
        Codec.build_vae_lora(self, cfg.vae_rank, cfg.vae_rank)
        Codec.build_DiT_lora(self, cfg.dit_rank, cfg.dit_rank, 32)

    @torch.no_grad()
    def analyze_image(self, x):
        aux = self.aux_codec((x + 1) / 2)
        latent = self.vae.encode(x).latent * self.vae.config.scaling_factor
        return latent, aux

    def render(self, z):
        return self.vae.decode(z / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

    def train(self, mode=True):
        super().train(mode)
        self.aux_codec.eval()
        vae=self.vae.get_base_model() if hasattr(self.vae,'get_base_model') else self.vae
        vae.encoder.eval()
        return self

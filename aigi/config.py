"""Strict configuration: typos and unsupported combinations never silently pass."""
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import json
import math


@dataclass
class Components:
    learned_token_weights: bool = True
    map_distillation: bool = True
    texture: bool = True
    importance: bool = True
    difficulty: bool = True
    explanation: bool = True
    correction: bool = True
    text_entropy: bool = True
    waterfill: bool = True
    weighted_loss: bool = True
    snr_time: bool = True
    semantic_time: bool = True
    time_residual: bool = True
    local_adaln: bool = True
    dual_guidance: bool = True
    self_distillation: bool = True
    latent_prompt: bool = True
    variance_flow: bool = True


@dataclass
class Config:
    method: str = 'joint'
    experiment: str = ''
    upstream_root: str = '.'
    upstream_config: str = 'configs/inference_merge.yaml'
    sana_path: str = 'aigi_assets/sana'
    weights: str = 'aigi_assets/ditic/q3_merge_ema.pt'
    elic_path: str = 'aigi_assets/ditic/elic_official.pth'
    checkpoint_format: str = 'merged'
    model_variant: str = 'fp16'
    offline: bool = True
    text_length: int = 77
    hidden: int = 128
    beta: float = .5
    quality: int = 3
    rate_weight: float = .5
    vae_rank: int = 64
    dit_rank: int = 128
    finetune: str = 'lora'
    dit_from_scratch: bool = False
    components: Components = field(default_factory=Components)
    train_manifest: str = 'aigi_data/train.jsonl'
    val_manifest: str = 'aigi_data/val.jsonl'
    teacher_cache: str = 'aigi_cache/teacher'
    maps_checkpoint: str = ''
    init_checkpoint: str = ''
    output: str = 'aigi_runs/joint_q3'
    stage: str = 'codec'
    crop: int = 256
    batch_size: int = 1
    accumulation: int = 1
    workers: int = 2
    steps: int = 100000
    lr: float = .0001
    aux_lr: float = .001
    optimizer: str = 'adamw'
    weight_decay: float = .01
    lr_gamma: float = .5
    gan_mode: str = 'adaptive'
    augmentation: str = 'center'
    views_per_image: int = 1
    seed: int = 903
    save_every: int = 1000
    val_every: int = 1000
    lpips_weight: float = 1.
    dists_weight: float = .2
    gan_weight: float = 0.
    gan_start_fraction: float = .3
    distill_weight: float = 1.
    waterfill_weight: float = .01
    ema_decay: float = .999
    deterministic: bool = True

    def validate(self):
        for name in ('quality','text_length','hidden','vae_rank','dit_rank','crop','batch_size','accumulation',
                     'workers','steps','views_per_image','seed','save_every','val_every'):
            if type(getattr(self,name)) is not int:raise TypeError(name+' must be an integer')
        for name in ('beta','rate_weight','lr','aux_lr','weight_decay','lr_gamma','lpips_weight','dists_weight',
                     'gan_weight','gan_start_fraction','distill_weight','waterfill_weight','ema_decay'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError(name+' must be finite numeric')
        if self.vae_rank < 1 or self.dit_rank < 1:raise ValueError('LoRA ranks must be positive')
        if self.optimizer not in {'adam','adamw'} or self.weight_decay < 0:
            raise ValueError('optimizer must be adam/adamw and weight_decay nonnegative')
        if not 0 < self.lr_gamma <= 1 or self.gan_mode not in {'adaptive','fixed'}:
            raise ValueError('Invalid learning-rate gamma or GAN mode')
        if self.augmentation not in {'center','cached_random'} or self.views_per_image < 1:
            raise ValueError('Invalid finite-view augmentation')
        if self.augmentation == 'center' and self.views_per_image != 1:
            raise ValueError('Center policy has exactly one view')
        for name in ('offline', 'deterministic', 'dit_from_scratch'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(name + ' must be boolean')
        for name in ('lr', 'aux_lr'):
            if getattr(self, name) <= 0:
                raise ValueError(name + ' must be positive')
        for name in ('lpips_weight', 'dists_weight', 'gan_weight', 'distill_weight', 'waterfill_weight'):
            if getattr(self, name) < 0:
                raise ValueError(name + ' must be nonnegative')
        if not 0 <= self.gan_start_fraction < 1:
            raise ValueError('gan_start_fraction must be in [0,1)')
        if self.method not in {'ditic', 'cadc_style', 'idea1', 'idea2', 'joint'}:
            raise ValueError('Unknown method: ' + self.method)
        if self.checkpoint_format not in {'merged', 'lora', 'base'}:
            raise ValueError('checkpoint_format must be merged/lora/base')
        if self.stage not in {'maps', 'codec'} or self.finetune not in {'lora', 'full', 'frozen'}:
            raise ValueError('Invalid stage or finetune')
        if self.crop < 256 or self.crop % 256:
            raise ValueError('Training crop must be a positive multiple of 256')
        for name in ('batch_size', 'accumulation', 'steps', 'save_every', 'val_every'):
            if getattr(self, name) < 1:
                raise ValueError(name + ' must be positive')
        if not 1 <= self.quality <= 4 or self.rate_weight <= 0:
            raise ValueError('quality must be 1..4 and rate_weight positive')
        if self.text_length != 77 or self.hidden < 8 or self.workers < 0:
            raise ValueError('This implementation fixes text_length=77; hidden>=8 and workers>=0')
        if not 0 <= self.beta <= 2 or not 0 <= self.ema_decay < 1:
            raise ValueError('beta must be in [0,2] and EMA decay in [0,1)')
        if self.dit_from_scratch and self.finetune != 'full':
            raise ValueError('Training a randomly initialized DiT requires finetune=full')
        for f in fields(Components):
            if type(getattr(self.components, f.name)) is not bool:
                raise TypeError('Component flag must be a boolean: ' + f.name)
        if self.stage == 'maps' and not self.components.map_distillation:
            raise ValueError('maps stage requires map_distillation=true')
        if self.method in {'ditic', 'cadc_style'} and self.stage == 'maps':
            raise ValueError('Original DiT-IC has no map-distillation stage')
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    import yaml
    d = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(d, dict):
        raise ValueError('Configuration must be an object')
    cs = d.pop('components', {})
    c = Config(**d, components=Components(**cs))
    return c.validate()


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')

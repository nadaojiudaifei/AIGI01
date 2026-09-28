"""Single GPU or torchrun DDP training with explicit FP32 numerical contract."""
from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from .data import ManifestDataset, collate, assert_disjoint
from .model import AIGICodec
from .losses import RateDistortionLoss
from .checkpoint import read_checkpoint, load_weights, save_training, EMA, rng_state, restore_rng
from .utils import seed_all, atomic_json, sha256_file
from .optimization import learning_rate, make_optimizer, adversarial_coefficient


def tensors_to(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def conditions(batch, device):
    if 'text' in batch:
        return tuple(batch[k].to(device) for k in ('text', 'mask', 'null_text', 'null_mask'))
    b = batch['image'].shape[0]
    z = torch.zeros(b, 77, 2304, device=device)
    m = torch.ones(b, 77, device=device, dtype=torch.bool)
    return z, m, z, m


def train(cfg, resume=None, device=None, stop_after=None):
    world, rank, local_rank = (int(os.environ.get(k, d)) for k, d in [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')])
    device = device or f'cuda:{local_rank}'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. This command requires the remote GPU environment; run cpu tests separately.')
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    try:
        return _train(cfg, resume, device, world, rank, local_rank, stop_after)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg, resume, device, world, rank, local_rank, stop_after=None):
    if stop_after is not None and not 1 <= stop_after <= cfg.steps:
        raise ValueError("stop_after must be an absolute optimizer step in [1,steps]")
    seed_all(cfg.seed + rank, cfg.deterministic)
    assert_disjoint(cfg.train_manifest, cfg.val_manifest)
    need_cache = cfg.method in {'idea1', 'idea2', 'joint'}
    train_data = ManifestDataset(cfg.train_manifest, cfg.crop, cfg.teacher_cache if need_cache else None,
                                 cfg.quality, require_cache=need_cache, augmentation=cfg.augmentation,
                                 views_per_image=cfg.views_per_image, seed=cfg.seed)
    valid_data = ManifestDataset(cfg.val_manifest, cfg.crop, cfg.teacher_cache if need_cache else None,
                                 cfg.quality, require_cache=need_cache)
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=cfg.seed, drop_last=True)
    generator = torch.Generator().manual_seed(cfg.seed + rank)
    loader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
                        num_workers=cfg.workers, collate_fn=collate, generator=generator, pin_memory=device.startswith('cuda'))
    if len(loader) == 0:
        raise ValueError('Not enough training images for batch_size x world_size')
    model = AIGICodec(cfg, device, training=True)
    if cfg.stage == 'maps':
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('entropy.student.', 'entropy.decoder_maps.')))
    elif cfg.maps_checkpoint and need_cache:
        load_weights(model, read_checkpoint(cfg.maps_checkpoint)['model'], maps_only=True)
    if cfg.init_checkpoint:
        load_weights(model, read_checkpoint(cfg.init_checkpoint)['model'])
    parameters = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith('.quantiles')]
    auxiliary = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith('.quantiles')]
    if not parameters:
        raise ValueError('No trainable parameters')
    optimizer = make_optimizer(parameters, cfg)
    aux_optimizer = torch.optim.Adam(auxiliary, lr=cfg.aux_lr) if auxiliary else None
    # Auxiliary quantiles are optimized outside DDP's forward graph.
    for p in auxiliary:
        p.requires_grad_(False)
    from .evaluate import network_guard
    with network_guard(cfg.offline):
        losses = RateDistortionLoss(cfg, device)
    discriminator = losses.discriminator
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(.5, .9)) if discriminator else None
    ema = EMA(model, cfg.ema_decay)
    start, resume_rng = 0, None
    if resume:
        state = read_checkpoint(resume)
        for key in ('method', 'stage', 'quality', 'crop', 'components', 'batch_size', 'accumulation', 'finetune', 'vae_rank', 'dit_rank',
                    'steps', 'lr', 'aux_lr', 'gan_weight', 'gan_start_fraction', 'lpips_weight', 'dists_weight',
                    'optimizer', 'weight_decay', 'lr_gamma', 'gan_mode', 'augmentation', 'views_per_image',
                    'rate_weight', 'beta', 'seed', 'train_manifest', 'val_manifest', 'teacher_cache'):
            if state['config'].get(key, getattr(cfg, key)) != cfg.to_dict()[key]:
                raise ValueError('Resume configuration mismatch: ' + key)
        for field in ('train_manifest', 'val_manifest'):
            expected = state.get('data_sha256', {}).get(field)
            if expected is None or sha256_file(getattr(cfg, field)) != expected:
                raise ValueError('Resume manifest content changed or legacy checkpoint lacks hash: '+field)
        load_weights(model, state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if aux_optimizer and state['aux_optimizer']:
            aux_optimizer.load_state_dict(state['aux_optimizer'])
        if discriminator and state['discriminator']:
            discriminator.load_state_dict(state['discriminator'])
            disc_optimizer.load_state_dict(state['disc_optimizer'])
        ema.shadow = state['ema'] or ema.shadow
        if len(state['rng']) != world:
            raise ValueError('Exact resume requires the same DDP world size')
        resume_rng = state['rng'][rank]
        start = int(state['step'])
    if start >= cfg.steps or (stop_after is not None and start >= stop_after):
        raise ValueError('Checkpoint is already at or beyond the requested stopping step')
    wrapped = DDP(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False) if world > 1 else model
    if discriminator and world > 1:
        discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)
    output = Path(cfg.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(cfg.to_dict(), output / 'resolved_config.json')
    consumed = start * cfg.accumulation
    epoch, offset = divmod(consumed, len(loader))
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    if resume_rng:
        restore_rng(resume_rng)
    for step in range(start, cfg.steps):
        begun = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        records, disc_batches = {}, []
        gan_active = discriminator is not None and step >= int(cfg.gan_start_fraction * cfg.steps)
        if discriminator:
            discriminator.requires_grad_(False)
            discriminator.eval()
        for micro in range(cfg.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = tensors_to(batch, device)
            context = wrapped.no_sync() if world > 1 and micro < cfg.accumulation - 1 else nullcontext()
            with context:
                out = wrapped(batch['image'], *conditions(batch, device))
                loss, logs = losses(out, batch, step)
                if gan_active:
                    disc_module = discriminator.module if isinstance(discriminator, DDP) else discriminator
                    generator_loss = -disc_module(out['image']).mean()
                    last_layer = model.backbone.codec.aux.block[-1].weight
                    coefficient = adversarial_coefficient(losses.reconstruction_objective, generator_loss, last_layer, cfg)
                    loss = loss + coefficient * generator_loss
                    logs['gan_coefficient'] = coefficient.detach()
                    logs['gan_g'] = generator_loss.detach()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at optimizer step {step}')
                (loss / cfg.accumulation).backward()
            for k, value in logs.items():
                records[k] = records.get(k, 0.) + value.detach() / cfg.accumulation
            if gan_active:
                disc_batches.append((batch['image'].detach(), out['image'].detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 4., error_if_nonfinite=True)
        optimizer.step()
        if aux_optimizer:
            for p in auxiliary:
                p.requires_grad_(True)
            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = model.backbone.codec.aux_loss()
            if not torch.isfinite(aux_loss):
                raise FloatingPointError('Non-finite entropy auxiliary loss')
            aux_loss.backward()
            # Quantile gradients bypass DDP forward; synchronize explicitly.
            if world > 1:
                for p in auxiliary:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world)
            aux_optimizer.step()
            for p in auxiliary:
                p.requires_grad_(False)
            records['aux_loss'] = aux_loss.detach()
        if gan_active:
            discriminator.requires_grad_(True)
            discriminator.train()
            disc_optimizer.zero_grad(set_to_none=True)
            disc_total = torch.zeros((),device=device)
            for micro,(real,fake) in enumerate(disc_batches):
                context = discriminator.no_sync() if world > 1 and micro < len(disc_batches)-1 else nullcontext()
                with context:
                    scores = discriminator(torch.cat([real, fake], 0))
                    real_scores, fake_scores = scores.chunk(2, 0)
                    disc_loss = .5 * (torch.relu(1-real_scores).mean()+torch.relu(1+fake_scores).mean())
                    if not torch.isfinite(disc_loss):
                        raise FloatingPointError('Non-finite discriminator loss')
                    (disc_loss/len(disc_batches)).backward()
                disc_total += disc_loss.detach()/len(disc_batches)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 4., error_if_nonfinite=True)
            disc_optimizer.step()
            records['gan_d'] = disc_total
        # Explicit piecewise LR schedule, independent of loader size.
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(cfg, step + 1)
        ema.update(model)
        records['grad_norm'] = norm.detach()
        if world > 1:
            for value in records.values():
                dist.all_reduce(value)
                value.div_(world)
        if rank == 0:
            record = {k: float(v) for k, v in records.items()}
            record.update(step=step + 1, seconds=time.perf_counter() - begun, lr=optimizer.param_groups[0]['lr'])
            with (output / 'train.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            if step == start or (step + 1) % 10 == 0:
                print(json.dumps(record), flush=True)
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            if world > 1:
                dist.barrier()
            if rank == 0:
                model.eval()
                vb = tensors_to(collate([valid_data[0]]), device)
                with torch.no_grad():
                    vo = model(vb['image'], *conditions(vb, device))
                    vl, _ = losses(vo, vb, step)
                atomic_json({'step': step + 1, 'validation_smoke_loss': float(vl),
                             'note': 'One deterministic validation crop. Use evaluate for the full native-resolution test set.'}, output / 'validation_smoke.json')
                model.train()
            if world > 1:
                dist.barrier()
        if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps or step + 1 == stop_after:
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(rng_state(), states, dst=0)
            else:
                states = [rng_state()]
            if rank == 0:
                disc_module = discriminator.module if isinstance(discriminator, DDP) else discriminator
                save_training(output / 'last.pt', model, cfg, step + 1, optimizer, aux_optimizer,
                              disc_module, disc_optimizer, ema.shadow, states)
            if world > 1:
                dist.barrier()
        if step + 1 == stop_after:
            break
    return str(output / 'last.pt')

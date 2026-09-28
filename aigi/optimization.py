"""Explicit optimizer/LR/GAN coefficients used in both training and tests."""
import torch


def learning_rate(cfg, completed_steps):
    if completed_steps < 0:
        raise ValueError('completed_steps must be nonnegative')
    decays = sum(completed_steps / cfg.steps >= p for p in (.5,.8,.9))
    return cfg.lr * cfg.lr_gamma ** decays


def make_optimizer(parameters, cfg):
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[cfg.optimizer]
    return cls(parameters, lr=cfg.lr, betas=(.9,.999), weight_decay=cfg.weight_decay)


def adversarial_coefficient(reconstruction_loss, adversarial_loss, last_layer, cfg):
    if cfg.gan_mode == 'fixed':
        return reconstruction_loss.new_tensor(cfg.gan_weight)
    # autograd.grad is used only to obtain the detached scalar; DDP still reduces
    # parameter gradients in loss.backward(), never here.
    rec = torch.autograd.grad(reconstruction_loss, last_layer, retain_graph=True)[0]
    adv = torch.autograd.grad(adversarial_loss, last_layer, retain_graph=True)[0]
    ratio = (rec.norm()/(adv.norm()+1e-4)).clamp(0,1e4).detach()
    return cfg.gan_weight * ratio

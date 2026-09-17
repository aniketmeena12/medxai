"""Attribution methods used in the paper (docs/04-preregistration.md §4).

- Grad-CAM (primary) on ``model.encoder.gradcam_layer``, identical for every model.
- RISE (secondary, gradient-free; Petsiuk et al., BMVC 2018).
- Model-parameter randomization sanity gate (Adebayo et al., NeurIPS 2018).

All functions return maps of shape (B, H_in, W_in), min-max normalized per image to [0, 1].
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _normalize(maps: torch.Tensor) -> torch.Tensor:
    flat = maps.flatten(1)
    lo, hi = flat.min(dim=1, keepdim=True).values, flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-12)).view_as(maps)


def _targets(logits: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    return logits.argmax(dim=1) if target is None else target


def gradcam(model: nn.Module, batch: dict, target: torch.Tensor | None = None) -> torch.Tensor:
    """Grad-CAM for class ``target`` (default: predicted class)."""
    layer = model.encoder.gradcam_layer
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_, __, out):
        out.retain_grad()
        store["act"] = out

    handle = layer.register_forward_hook(fwd_hook)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            image = batch["image"].requires_grad_(False)
            logits = model(image, batch.get("cat"), batch.get("cont"), batch.get("cont_mask"))
            cls = _targets(logits, target)
            model.zero_grad(set_to_none=True)
            logits.gather(1, cls[:, None]).sum().backward()
        act, grad = store["act"], store["act"].grad
        if act.ndim == 3:  # ViT tokens -> grid
            n_prefix = act.shape[1] - int(np.sqrt(act.shape[1])) ** 2
            side = int(np.sqrt(act.shape[1] - n_prefix))
            act = act[:, n_prefix:].transpose(1, 2).reshape(act.shape[0], -1, side, side)
            grad = grad[:, n_prefix:].transpose(1, 2).reshape(grad.shape[0], -1, side, side)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    finally:
        handle.remove()
        model.train(was_training)
    return _normalize(cam[:, 0].detach())


@torch.no_grad()
def rise(
    model: nn.Module,
    batch: dict,
    target: torch.Tensor | None = None,
    n_masks: int = 2000,
    grid: int = 7,
    p_keep: float = 0.5,
    chunk: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """RISE saliency. Metadata is kept fixed while the image is randomly masked."""
    was_training = model.training
    model.eval()
    image = batch["image"]
    b, _, h, w = image.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cell_h, cell_w = int(np.ceil(h / grid)), int(np.ceil(w / grid))
    coarse = (torch.rand(n_masks, 1, grid, grid, generator=gen) < p_keep).float()
    up = F.interpolate(coarse, size=((grid + 1) * cell_h, (grid + 1) * cell_w), mode="bilinear",
                       align_corners=False)
    dx = torch.randint(0, cell_w, (n_masks,), generator=gen)
    dy = torch.randint(0, cell_h, (n_masks,), generator=gen)
    masks = torch.stack([up[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n_masks)])
    masks = masks.to(image.device)

    logits = model(image, batch.get("cat"), batch.get("cont"), batch.get("cont_mask"))
    cls = _targets(logits, target)
    sal = torch.zeros(b, h, w, device=image.device)
    for i in range(b):
        meta = {k: batch[k][i : i + 1] for k in ("cat", "cont", "cont_mask") if k in batch}
        for s in range(0, n_masks, chunk):
            m = masks[s : s + chunk]
            x = image[i : i + 1] * m
            rep = {k: v.expand(len(m), *v.shape[1:]) for k, v in meta.items()}
            probs = model(x, rep.get("cat"), rep.get("cont"), rep.get("cont_mask")).softmax(1)
            sal[i] += (probs[:, cls[i]][:, None, None] * m[:, 0]).sum(0)
    model.train(was_training)
    return _normalize(sal / (n_masks * p_keep))


def randomized_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    """Copy of ``model`` with every parameter re-drawn from N(0, 0.02) (full randomization)."""
    clone = copy.deepcopy(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in clone.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    return clone


def ssim_global(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Single-window SSIM between batches of [0, 1] maps (B, H, W) -> (B,)."""
    a, b = a.flatten(1), b.flatten(1)
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(1), b.mean(1)
    var_a, var_b = a.var(1), b.var(1)
    cov = ((a - mu_a[:, None]) * (b - mu_b[:, None])).mean(1)
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2) + eps)


def passes_sanity_gate(maps_trained: torch.Tensor, maps_random: torch.Tensor,
                       threshold: float = 0.5) -> tuple[bool, float]:
    """Pre-registered gate: the method fails if mean SSIM(trained, randomized) >= threshold."""
    score = float(ssim_global(maps_trained, maps_random).mean())
    return score < threshold, score

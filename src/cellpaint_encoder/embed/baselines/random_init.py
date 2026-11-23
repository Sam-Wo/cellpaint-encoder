"""b6: random-init ViT-S/16 (5 channels), a floor. Seeded so the baseline is reproducible."""

from __future__ import annotations

import torch

from ...models.vit import build_vit
from ..encoders import ViTEncoder


def build(cfg, mean, std, device="cpu", seed: int = 0, **kw) -> ViTEncoder:
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    vit = build_vit(cfg.model, drop_path=0.0)
    torch.random.set_rng_state(g)
    return ViTEncoder("random_init", vit, mean, std, device, **kw)

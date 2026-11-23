"""b5: ImageNet DINO ViT-S/16, frozen, patch embedding inflated to 5 channels (our initialisation)."""

from __future__ import annotations

from ...models.vit import build_vit, load_imagenet_dino
from ..encoders import ViTEncoder


def build(cfg, mean, std, device="cpu", **kw) -> ViTEncoder:
    vit = build_vit(cfg.model, drop_path=0.0)
    load_imagenet_dino(vit, cfg.model.init_weights)
    return ViTEncoder("imagenet_dino", vit, mean, std, device, **kw)

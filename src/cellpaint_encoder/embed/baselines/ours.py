"""Our trained encoder: the EMA teacher backbone of a run (``teacher_backbone.pt`` or a checkpoint)."""

from __future__ import annotations

import torch

from ...config import from_dict
from ...models.vit import build_vit
from ..encoders import ViTEncoder


def load_backbone(path: str):
    st = torch.load(path, map_location="cpu", weights_only=True)
    cfg = from_dict(st["config"])
    vit = build_vit(cfg.model, drop_path=0.0)
    if "backbone" in st:
        sd = st["backbone"]
    else:  # training checkpoint
        sd = {k[len("backbone.") :]: v for k, v in st["teacher"].items() if k.startswith("backbone.")}
    vit.load_state_dict(sd, strict=True)
    return vit, cfg, st["channel_mean"], st["channel_std"]


def build(path: str, device="cpu", name="ours", **kw) -> ViTEncoder:
    vit, _, mean, std = load_backbone(path)
    return ViTEncoder(name, vit, mean, std, device, **kw)

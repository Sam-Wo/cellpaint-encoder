"""Encoder interface shared by our model and every image baseline.

An encoder declares its crop and grid, a ``preprocess`` that turns one decoded site into tiles
(each model has its **own** preprocessing adapter, DESIGN.md 1.3), and ``embed`` that maps a
tile batch to per-tile vectors (CLS token, 4.6 step 2). Site embedding = mean over tiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.normalize import normalise_site
from .grids import grid_tiles


@dataclass
class SiteContext:
    """What a preprocessing adapter may use: raw uint16, illum-corrected float32, plate percentiles."""

    raw: np.ndarray  # uint16 [5, H, W], canonical channel order
    corrected: np.ndarray | None  # float32 [5, H, W] after illumination correction
    plate_norm: dict | None
    meta: dict = field(default_factory=dict)


class Encoder:
    name = "encoder"
    crop = 224
    grid = 4
    window: int | None = None

    def preprocess(self, ctx: SiteContext) -> np.ndarray:
        raise NotImplementedError

    def embed(self, tiles: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ViTEncoder(Encoder):
    """Ours, ImageNet-frozen and random-init: per-plate DMSO uint8 -> /255 -> channel z -> CLS."""

    def __init__(
        self,
        name: str,
        backbone: torch.nn.Module,
        mean,
        std,
        device="cpu",
        crop=224,
        grid=4,
        window=None,
        pct=(0.1, 99.9),
        norm_mode="per_plate_dmso",
        amp: bool = False,
    ):
        self.name, self.crop, self.grid, self.window = name, crop, grid, window
        self.backbone = backbone.eval().to(device)
        self.device = torch.device(device)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self.pct, self.norm_mode, self.amp = pct, norm_mode, amp

    def preprocess(self, ctx: SiteContext) -> np.ndarray:
        u8 = normalise_site(ctx.corrected, self.norm_mode, ctx.plate_norm, *self.pct)
        return grid_tiles(u8, self.crop, self.grid, self.window)

    def preprocess_u8(self, u8: np.ndarray) -> np.ndarray:  # dev cache: already normalised
        return grid_tiles(u8, self.crop, self.grid, self.window)

    @torch.no_grad()
    def embed(self, tiles: torch.Tensor) -> torch.Tensor:
        x = (tiles.to(self.device).float() / 255.0 - self.mean) / self.std
        ctx = (
            torch.autocast(self.device.type, dtype=torch.bfloat16)
            if self.amp
            else torch.autocast("cpu", enabled=False)
        )
        with ctx:
            return self.backbone(x).float()

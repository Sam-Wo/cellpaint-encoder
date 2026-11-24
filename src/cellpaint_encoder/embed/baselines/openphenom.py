"""b3: Recursion OpenPhenom (CA-MAE ViT-S/16, channel-agnostic) adapter. **Recursion Non-Commercial
EULA: never redistribute weights or embeddings.**

Documented interface (Hugging Face model card / repository): ``MAEModel.from_pretrained(dir)``
from the repository's ``huggingface_mae.py``; ``model.predict(x)`` takes a uint8 tensor
``[N, C, 256, 256]`` (any number of channels) and returns ``[N, 384]``; the model applies its own
normaliser internally. Loading it executes the repository's Python code, so do it only in the
sandbox (RUNBOOK.md). Until the fidelity gate G-B: how the uint16 site becomes uint8 (default: a
per-image 0.1/99.9 percentile stretch of the raw image), and whether illumination correction is
applied, are parameters.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import torch

from ...data import resample
from ..encoders import Encoder, SiteContext
from ..grids import grid_tiles


def to_uint8_per_image(site: np.ndarray, lo: float = 0.1, hi: float = 99.9, scale: float = 1.0) -> np.ndarray:
    """Per-image percentile stretch at native resolution, then the R7 resample, clip, round."""
    flat = site.reshape(site.shape[0], -1)
    p_lo, p_hi = np.percentile(flat, [lo, hi], axis=1)
    x = (site - p_lo[:, None, None]) / np.maximum(p_hi - p_lo, 1e-6)[:, None, None]
    x = resample.resample_site(x, scale)
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


class OpenPhenomEncoder(Encoder):
    def __init__(
        self,
        model,
        device="cpu",
        use_illum: bool = False,
        pct=(0.1, 99.9),
        channel_order=(0, 1, 2, 3, 4),
        crop=256,
        grid=3,
        window=None,
    ):
        self.name, self.crop, self.grid, self.window = "openphenom", crop, grid, window
        self.model, self.device = model, torch.device(device)
        self.use_illum, self.pct, self.order = use_illum, pct, list(channel_order)

    def preprocess(self, ctx: SiteContext) -> np.ndarray:
        site = ctx.corrected if self.use_illum else ctx.raw.astype(np.float32)
        u8 = to_uint8_per_image(site[self.order], *self.pct, scale=ctx.scale)
        return grid_tiles(u8, self.crop, self.grid, self.window)

    @torch.no_grad()
    def embed(self, tiles: torch.Tensor) -> torch.Tensor:
        if tiles.dtype != torch.uint8:
            raise TypeError("OpenPhenom expects uint8 tiles")
        return torch.as_tensor(self.model.predict(tiles.to(self.device))).float()


def build(repo_dir: str, weights_dir: str | None = None, device="cpu", **kw) -> OpenPhenomEncoder:
    """Sandbox only: imports ``huggingface_mae`` from the cloned model repository."""
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    mod = importlib.import_module("huggingface_mae")
    model = mod.MAEModel.from_pretrained(weights_dir or repo_dir).to(device).eval()
    return OpenPhenomEncoder(model, device, **kw)

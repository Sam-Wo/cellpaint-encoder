"""b1: DeepCellMorpho (J&J; Abdelmoula et al., Nature Methods 2026) adapter. **Weights are CC BY-NC-SA
4.0 and pickle-based: never download them into the main environment and never redistribute.**

Interface documented in their repository (DESIGN.md 2): DINO / DINOv2 ViT-B/16 (and a DINOv2
ViT-S/16) on 480x480x5 inputs; per-image percentile clipping then min-max per channel
(``models/dinos/util/datasets.py``); CLS embedding. Unknown until the fidelity gate G-B and
therefore parameters here, not constants: the clip percentiles, whether illumination correction
is applied, the channel order, and any mean/std after min-max.

Loading: run ``scripts/convert_untrusted_checkpoint.py`` **inside a throwaway container** to turn
their ``.pth`` into a plain tensor dict; this module only ever calls
``torch.load(weights_only=True)`` on that converted file.
"""

from __future__ import annotations

import numpy as np
import torch

from ...data import resample
from ...models.vit import VisionTransformer
from ..encoders import Encoder, SiteContext
from ..grids import grid_tiles

ARCHS = {
    "dino_vitb16": dict(embed_dim=768, depth=12, num_heads=12, layerscale=False),
    "dinov2_vitb16": dict(embed_dim=768, depth=12, num_heads=12, layerscale=True),
    "dinov2_vits16": dict(embed_dim=384, depth=12, num_heads=6, layerscale=True),
}
PREFIXES = ("module.", "teacher.", "student.", "backbone.", "encoder.")
DROP = ("head", "dino_head", "ibot_head", "mask_token", "register_tokens")


def convert_state_dict(sd: dict) -> dict:
    """Strip wrapper prefixes and drop projection heads / mask tokens."""
    if "teacher" in sd and isinstance(sd["teacher"], dict):
        sd = sd["teacher"]
    elif "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = {}
    for k, v in sd.items():
        for p in PREFIXES:
            while k.startswith(p):
                k = k[len(p) :]
        if k.split(".")[0] in DROP:
            continue
        out[k] = v
    return out


def build_model(
    arch: str, state_dict: dict, in_chans: int = 5, img_size: int | None = None
) -> VisionTransformer:
    a = ARCHS[arch]
    sd = convert_state_dict(state_dict)
    n_pos = sd["pos_embed"].shape[1] - 1
    side = int(round(n_pos**0.5))
    vit = VisionTransformer(
        img_size=img_size or side * 16,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=a["embed_dim"],
        depth=a["depth"],
        num_heads=a["num_heads"],
        layerscale=a["layerscale"],
    )
    res = vit.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise KeyError(
            f"DeepCellMorpho {arch}: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}"
        )
    return vit


def per_image_clip_minmax(site: np.ndarray, lo: float, hi: float) -> np.ndarray:
    flat = site.reshape(site.shape[0], -1)
    p_lo, p_hi = np.percentile(flat, [lo, hi], axis=1)
    x = np.clip(site, p_lo[:, None, None], p_hi[:, None, None])
    return ((x - p_lo[:, None, None]) / np.maximum(p_hi - p_lo, 1e-6)[:, None, None]).astype(np.float32)


class DeepCellMorphoEncoder(Encoder):
    def __init__(
        self,
        model: torch.nn.Module,
        arch: str = "dinov2_vitb16",
        device="cpu",
        clip_pct=(1.0, 99.0),
        use_illum: bool = False,
        channel_order: tuple[int, ...] = (0, 1, 2, 3, 4),
        input_mean=None,
        input_std=None,
        crop: int = 480,
        grid: int = 2,
        window=None,
    ):
        self.name, self.crop, self.grid, self.window = f"deepcellmorpho_{arch}", crop, grid, window
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.clip_pct, self.use_illum, self.order = clip_pct, use_illum, list(channel_order)
        self.mean = (
            None if input_mean is None else torch.tensor(input_mean, device=self.device).view(1, -1, 1, 1)
        )
        self.std = (
            None if input_std is None else torch.tensor(input_std, device=self.device).view(1, -1, 1, 1)
        )

    def preprocess(self, ctx: SiteContext) -> np.ndarray:
        site = ctx.corrected if self.use_illum else ctx.raw.astype(np.float32)
        # R7: their normalisation is per 480^2 crop, so it cannot precede the crop grid; resample
        # the (linear) float input instead, then crop at the common pixel size and normalise
        site = resample.resample_site(site[self.order], ctx.scale)
        tiles = grid_tiles(site, self.crop, self.grid, self.window)
        # their normalisation is per image = per 480^2 crop
        return np.stack([per_image_clip_minmax(t, *self.clip_pct) for t in tiles])

    @torch.no_grad()
    def embed(self, tiles: torch.Tensor) -> torch.Tensor:
        x = tiles.to(self.device).float()
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return self.model(x).float()


def build(converted_path: str, arch: str = "dinov2_vitb16", device="cpu", **kw) -> DeepCellMorphoEncoder:
    sd = torch.load(converted_path, map_location="cpu", weights_only=True)
    return DeepCellMorphoEncoder(build_model(arch, sd), arch, device, **kw)

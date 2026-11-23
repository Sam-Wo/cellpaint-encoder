"""Per-model crop grids (DESIGN.md 4.6, step 1): each model's native crop on the largest centred
grid that fits the site; optionally restricted to a common central window (768^2 sensitivity).

| model                      | crop | grid on a ~1000^2 site | grid inside the 768^2 window |
|----------------------------|------|------------------------|------------------------------|
| ours / ImageNet / random   | 224  | 4x4 (896^2)            | 3x3 (672^2)                  |
| DeepCellMorpho DINO/DINOv2 | 480  | 2x2 (960^2)            | 1x1                          |
| OpenPhenom                 | 256  | 3x3 (768^2)            | 3x3                          |
"""

from __future__ import annotations

import numpy as np

from ..data.tiles import centre_crop, centred_grid

GRIDS = {
    "ours": (4, 224),
    "imagenet_dino": (4, 224),
    "random_init": (4, 224),
    "deepcellmorpho": (2, 480),
    "openphenom": (3, 256),
}


def grid_tiles(site: np.ndarray, crop: int, n: int | None = None, window: int | None = None) -> np.ndarray:
    """``site [C, H, W]`` -> ``[n*n, C, crop, crop]`` from the centred grid.

    ``n=None`` takes as many tiles as fit; ``window`` first restricts the site to its centre."""
    if window:
        site = centre_crop(site, window)
    H, W = site.shape[-2:]
    if n is not None and n * crop > min(H, W):
        n = min(H, W) // crop
    corners = centred_grid(H, W, crop, n if n is not None else min(H // crop, W // crop))
    return np.stack([site[:, y : y + crop, x : x + crop] for y, x in corners])

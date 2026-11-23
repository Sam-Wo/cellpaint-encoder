"""Per-plate DMSO intensity normalisation -> uint8 (DESIGN.md 1.3, 3.6).

Each channel of a plate is scaled with the 0.1st and 99.9th percentiles of the
illumination-corrected DMSO sites of that plate (``data.norm_sites_per_plate`` sites, pooled
pixels). Differences in intensity *between wells* survive, which is biology; per-image min-max
would remove them. ``per_image`` mode (ablation A2) uses each image's own percentiles.
"""

from __future__ import annotations

import numpy as np

MAX_PIX_PER_SITE = 250_000  # pixel subsample per site and channel when pooling percentiles


def plate_percentiles(sites: list[np.ndarray], lo: float = 0.1, hi: float = 99.9,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``sites``: list of float32 ``[C, H, W]``. Returns (lo[C], hi[C]) over pooled pixels."""
    if not sites:
        raise ValueError("no DMSO sites to compute plate percentiles")
    rng = np.random.default_rng(seed)
    C = sites[0].shape[0]
    pooled = [[] for _ in range(C)]
    for s in sites:
        flat = s.reshape(C, -1)
        n = flat.shape[1]
        idx = rng.choice(n, size=min(n, MAX_PIX_PER_SITE), replace=False) if n > MAX_PIX_PER_SITE else slice(None)
        for c in range(C):
            pooled[c].append(flat[c, idx])
    p_lo = np.array([np.percentile(np.concatenate(p), lo) for p in pooled], np.float32)
    p_hi = np.array([np.percentile(np.concatenate(p), hi) for p in pooled], np.float32)
    bad = ~(p_hi > p_lo)
    p_hi[bad] = p_lo[bad] + 1e-6
    return p_lo, p_hi


def image_percentiles(site: np.ndarray, lo: float = 0.1, hi: float = 99.9) -> tuple[np.ndarray, np.ndarray]:
    flat = site.reshape(site.shape[0], -1)
    p_lo, p_hi = np.percentile(flat, [lo, hi], axis=1).astype(np.float32)
    bad = ~(p_hi > p_lo)
    p_hi[bad] = p_lo[bad] + 1e-6
    return p_lo, p_hi


def to_uint8(site: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray) -> np.ndarray:
    lo = p_lo[:, None, None]
    scale = 255.0 / (p_hi - p_lo)[:, None, None]
    x = (site - lo) * scale
    np.clip(x, 0, 255, out=x)
    return np.rint(x).astype(np.uint8)


def normalise_site(site: np.ndarray, mode: str, plate_norm: dict | None, lo: float, hi: float) -> np.ndarray:
    if mode == "per_plate_dmso":
        if plate_norm is None:
            raise KeyError("per_plate_dmso normalisation needs the plate's percentiles")
        return to_uint8(site, np.asarray(plate_norm["lo"], np.float32), np.asarray(plate_norm["hi"], np.float32))
    if mode == "per_image":
        return to_uint8(site, *image_percentiles(site, lo, hi))
    raise ValueError(f"unknown norm_mode {mode!r}")

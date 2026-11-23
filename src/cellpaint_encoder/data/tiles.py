"""Tiling and cropping (DESIGN.md 1.2, 4.6).

Training: 2 tiles of 448^2 per site, centre-jittered, kept only if the Otsu DNA foreground
covers at least 1% of the tile (Kim et al.). Evaluation: each model's own crop on the largest
centred grid that fits (4x4 of 224 for ours).
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(img: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold of a uint8 (or any) image via its histogram."""
    x = img.ravel()
    if img.dtype == np.uint8:
        hist = np.bincount(x, minlength=256).astype(np.float64)
        centers = np.arange(256, dtype=np.float64)
    else:
        hist, edges = np.histogram(x, bins=nbins)
        hist = hist.astype(np.float64)
        centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1e-12)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1
    return float(centers[int(np.argmax(between))])


def centred_grid(h: int, w: int, tile: int, n: int | None = None) -> list[tuple[int, int]]:
    """Top-left corners of the largest centred ``n x n`` (default: as many as fit) grid of tiles."""
    ny, nx = (h // tile, w // tile) if n is None else (n, n)
    if ny * tile > h or nx * tile > w or ny < 1 or nx < 1:
        raise ValueError(f"{ny}x{nx} grid of {tile} does not fit a {h}x{w} site")
    y0, x0 = (h - ny * tile) // 2, (w - nx * tile) // 2
    return [(y0 + i * tile, x0 + j * tile) for i in range(ny) for j in range(nx)]


def centre_crop(site: np.ndarray, size: int) -> np.ndarray:
    h, w = site.shape[-2:]
    if h < size or w < size:
        raise ValueError(f"site {h}x{w} smaller than {size}")
    y0, x0 = (h - size) // 2, (w - size) // 2
    return site[..., y0 : y0 + size, x0 : x0 + size]


def training_tiles(site_u8: np.ndarray, tile: int, n_tiles: int, fg_min: float,
                   rng: np.random.Generator) -> tuple[list[np.ndarray], dict]:
    """Up to ``n_tiles`` tiles from the cells of a centred grid, each jittered inside the slack.

    Returns (tiles, info) where info counts candidates rejected by the foreground filter."""
    C, H, W = site_u8.shape
    cells = centred_grid(H, W, tile)
    ny, nx = H // tile, W // tile
    slack_y, slack_x = (H - ny * tile) // 2, (W - nx * tile) // 2
    thr = otsu_threshold(site_u8[0])
    fg = site_u8[0] > thr
    order = rng.permutation(len(cells))
    tiles, rejected, fracs = [], 0, []
    for k in order:
        y, x = cells[k]
        y = int(np.clip(y + rng.integers(-slack_y, slack_y + 1), 0, H - tile)) if slack_y else y
        x = int(np.clip(x + rng.integers(-slack_x, slack_x + 1), 0, W - tile)) if slack_x else x
        frac = float(fg[y : y + tile, x : x + tile].mean())
        fracs.append(frac)
        if frac < fg_min:
            rejected += 1
            continue
        tiles.append(np.ascontiguousarray(site_u8[:, y : y + tile, x : x + tile]))
        if len(tiles) == n_tiles:
            break
    return tiles, {"rejected": rejected, "fg_fracs": fracs, "otsu": thr}

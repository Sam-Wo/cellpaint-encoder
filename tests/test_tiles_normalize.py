import numpy as np
import pytest

from cellpaint_encoder.data.normalize import normalise_site, plate_percentiles, to_uint8
from cellpaint_encoder.data.tiles import centre_crop, centred_grid, otsu_threshold, training_tiles
from cellpaint_encoder.embed.grids import grid_tiles


def test_otsu_bimodal():
    x = np.concatenate([np.full(1000, 20), np.full(1000, 200)]).astype(np.uint8)
    t = otsu_threshold(x)
    assert 20 <= t < 200


def test_centred_grid_and_crop():
    g = centred_grid(996, 996, 224)
    assert len(g) == 16 and g[0] == (50, 50)
    with pytest.raises(ValueError):
        centred_grid(100, 100, 224)
    assert centre_crop(np.zeros((5, 1000, 1000)), 896).shape == (5, 896, 896)


def test_grid_tiles_per_model():
    site = np.zeros((5, 996, 996), np.uint8)
    assert grid_tiles(site, 224, 4).shape == (16, 5, 224, 224)
    assert grid_tiles(site, 480, 2).shape == (4, 5, 480, 480)
    assert grid_tiles(site, 256, 3).shape == (9, 5, 256, 256)
    assert grid_tiles(site, 224, 4, window=768).shape == (9, 5, 224, 224)  # 3x3 inside 768


def test_foreground_filter_and_jitter():
    rng = np.random.default_rng(0)
    site = np.zeros((5, 996, 996), np.uint8)
    site[0, :448, :448] = rng.integers(150, 255, (448, 448))  # cells only in the top-left cell
    tiles, info = training_tiles(site, 448, 2, 0.01, rng)
    assert len(tiles) == 1 and info["rejected"] >= 1
    assert tiles[0].shape == (5, 448, 448)
    full = rng.integers(0, 255, (5, 996, 996), dtype=np.uint8)
    t2, _ = training_tiles(full, 448, 2, 0.01, np.random.default_rng(1))
    assert len(t2) == 2


def test_plate_percentiles_and_uint8():
    rng = np.random.default_rng(0)
    sites = [rng.normal(100, 10, (5, 64, 64)).astype(np.float32) for _ in range(4)]
    lo, hi = plate_percentiles(sites, 0.1, 99.9)
    assert lo.shape == (5,) and (hi > lo).all()
    u8 = to_uint8(sites[0], lo, hi)
    assert u8.dtype == np.uint8 and u8.min() <= 5 and u8.max() >= 250


def test_per_plate_keeps_between_well_intensity_but_per_image_does_not():
    base = np.random.default_rng(0).random((5, 64, 64)).astype(np.float32)
    bright = base * 2
    norm = {"lo": [0.0] * 5, "hi": [2.0] * 5}
    a = normalise_site(base, "per_plate_dmso", norm, 0.1, 99.9)
    b = normalise_site(bright, "per_plate_dmso", norm, 0.1, 99.9)
    assert b.mean() > 1.8 * a.mean()  # brighter well stays brighter (biology kept)
    c = normalise_site(base, "per_image", None, 0.1, 99.9)
    d = normalise_site(bright, "per_image", None, 0.1, 99.9)
    assert abs(int(c.mean()) - int(d.mean())) <= 1  # per-image removes it
    with pytest.raises(KeyError):
        normalise_site(base, "per_plate_dmso", None, 0.1, 99.9)

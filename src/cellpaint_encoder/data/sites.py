"""Read one site: 5 fluorescence TIFFs -> illumination-corrected float32 ``[5, H, W]``.

The same decode path feeds the tile cache (``write_shards``) and the embedding pass
(``embed``): each site is decoded once as raw uint16 per channel (DESIGN.md 1.3).
"""

from __future__ import annotations

import threading

import numpy as np

from .. import CHANNELS
from .gallery import Gallery, illum_key
from .illum import apply_illum, load_illum, read_npy_header
from .tiff_io import decode_tiff, is_skipped_file


class IllumCache:
    """Per-plate illumination functions, loaded once per plate (thread-safe)."""

    def __init__(self, gallery: Gallery, max_plates: int | None = None):
        self.g = gallery
        self.max_plates = max_plates  # keep only the most recent plates (worker processes)
        self._d: dict[tuple[str, str], np.ndarray] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self.header_info: dict[str, dict] = {}

    def get(
        self,
        source: str,
        batch: str,
        plate: str,
        channel: str,
        key: str | None = None,
        shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        k = (plate, channel)
        with self._lock:
            if k in self._d:
                return self._d[k]
            kl = self._key_locks.setdefault(k, threading.Lock())
        with kl:  # one load per (plate, channel), even with many threads
            with self._lock:
                if k in self._d:
                    return self._d[k]
            data = self.g.get(key or illum_key(source, batch, plate, channel))
            dt, shp, _ = read_npy_header(data)
            arr = load_illum(data, expect_shape=shape)
            with self._lock:
                self._d[k] = arr
                if self.max_plates:
                    plates = list(dict.fromkeys(p for p, _ in self._d))
                    for old in plates[: max(0, len(plates) - self.max_plates)]:
                        for kk in [kk for kk in self._d if kk[0] == old]:
                            del self._d[kk]
                self.header_info[f"{source}/{plate}/{channel}"] = {"dtype": str(dt), "shape": list(shp)}
        return arr


def read_raw_site(g: Gallery, row) -> np.ndarray:
    """uint16 ``[5, H, W]`` in canonical channel order."""
    chans = []
    for c in CHANNELS:
        key = getattr(row, f"key_{c}")
        if is_skipped_file(key):
            raise ValueError(f"refusing calibration/thumbnail file {key}")
        chans.append(decode_tiff(g.get(key)))
    shapes = {a.shape for a in chans}
    if len(shapes) != 1:
        raise ValueError(f"channel shapes differ within a site: {shapes}")
    return np.stack(chans).astype(np.uint16, copy=False)


def correct_site(raw: np.ndarray, illum: IllumCache, row) -> np.ndarray:
    out = np.empty(raw.shape, np.float32)
    for i, c in enumerate(CHANNELS):
        key = getattr(row, f"illum_{c}", None)
        if not isinstance(key, str) or not key:
            key = None
        fn = illum.get(row.source, row.batch, row.plate, c, key, shape=raw.shape[1:])
        out[i] = apply_illum(raw[i], fn)
    return out


def read_site(g: Gallery, illum: IllumCache | None, row) -> np.ndarray:
    raw = read_raw_site(g, row)
    return correct_site(raw, illum, row) if illum is not None else raw.astype(np.float32)


def read_site_uint8(g: Gallery, illum: IllumCache, row, lo, hi) -> np.ndarray:
    """Fused decode -> illum divide -> plate percentile scaling -> uint8, one channel at a time.

    Same result as ``normalise_site(read_site(...), "per_plate_dmso", ...)`` up to float rounding,
    but each channel (~4 MB float32) stays in CPU cache, so the cache build is not memory-bound
    when many worker processes run at once."""
    out = None
    for i, c in enumerate(CHANNELS):
        key = getattr(row, f"key_{c}")
        if is_skipped_file(key):
            raise ValueError(f"refusing calibration/thumbnail file {key}")
        raw = decode_tiff(g.get(key))
        ikey = getattr(row, f"illum_{c}", None)
        fn = illum.get(
            row.source,
            row.batch,
            row.plate,
            c,
            ikey if isinstance(ikey, str) and ikey else None,
            shape=raw.shape,
        )
        if out is None:
            out = np.empty((len(CHANNELS), *raw.shape), np.uint8)
        elif raw.shape != out.shape[1:]:
            raise ValueError("channel shapes differ within a site")
        x = raw.astype(np.float32)
        x /= fn
        x -= np.float32(lo[i])
        x *= np.float32(255.0 / (hi[i] - lo[i]))
        np.clip(x, 0, 255, out=x)
        np.rint(x, out=x)
        out[i] = x
    return out

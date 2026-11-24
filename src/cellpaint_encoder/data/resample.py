"""Resample sites to a common pixel size (DESIGN.md risk R7).

Pixel sizes differ between sources (0.598 to 0.703 µm/px across the sources we use, a 17.6%
spread, above the 15% threshold), so resolution would become a source signal. Every path that
produces pixels for a model (the training cache, the dev cache, the per-image ablation cache and
``embed`` for every encoder) calls :func:`resample_site`. A site's scale factor is
``native_um_per_px / data.resample_um_per_px``; the new size is ``round(old_size * factor)``.
A source whose ``|factor - 1| <= data.resample_tol`` keeps its native pixels (the function then
returns its input unchanged), so the held-out source_6 stays untouched.

Order (identical in every path): illumination correction and the intensity normalisation's affine
map are computed at native resolution (plate percentiles too), then the float image is resampled,
then clipped and rounded. Antialiased bilinear interpolation is linear with non-negative weights,
so it commutes with the affine map: the result equals "resample the corrected image, then
normalise", and only clipping and rounding happen after it.

Interpolation: ``torch.nn.functional.interpolate(mode="bilinear", antialias=True,
align_corners=False)`` on CPU float32, the same call for down- (0.92) and upsampling (1.082). Always
on the CPU so that a CPU-built cache and a GPU embedding pass see the same pixels.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

# jump-cellpainting/datasets metadata/microscope_config.csv at commit
# b3b4555d44cac440ffda7e25f3adbeb34fbee7fe, column Metadata_Pixel_Size_Microns. source_8 is also
# confirmed by its MetaSeries TIFF description (spatial-calibration-x 0.7032 µm, binning 2x2).
# The configs carry the same table as ``data.pixel_size_um``; tests check they agree.
PIXEL_SIZE_UM = {
    "source_1": 0.597976,
    "source_2": 0.65,
    "source_3": 0.597976,
    "source_4": 0.597976,
    "source_5": 0.64974,
    "source_6": 0.646187,
    "source_7": 0.65,
    "source_8": 0.7032,
    "source_9": 0.597976,
    "source_10": 0.6516944,
    "source_11": 0.597976,
    "source_13": 0.65,
    "source_15": 0.597976,
}


def source_scale(
    source: str, target_um_per_px: float | None, tol: float, pixel_size_um: Mapping[str, float]
) -> float:
    """Scale factor for one source: 1.0 when resampling is off or the source is within ``tol``."""
    if target_um_per_px is None:
        return 1.0
    if source not in pixel_size_um:
        raise KeyError(
            f"data.resample_um_per_px is set but {source!r} has no entry in data.pixel_size_um; "
            "add its pixel size (microscope_config.csv) before caching or embedding it"
        )
    f = float(pixel_size_um[source]) / float(target_um_per_px)
    return 1.0 if abs(f - 1.0) <= tol else f


def scale_for(dc, source: str) -> float:
    """Scale factor of ``source`` under a ``DataCfg``."""
    return source_scale(source, dc.resample_um_per_px, dc.resample_tol, dc.pixel_size_um)


def resampled_shape(h: int, w: int, scale: float) -> tuple[int, int]:
    return int(math.floor(h * scale + 0.5)), int(math.floor(w * scale + 0.5))


def resample_site(x: np.ndarray, scale: float) -> np.ndarray:
    """Resample a float image ``[H, W]`` or ``[C, H, W]`` (each channel separately) by ``scale``.

    ``scale == 1.0`` returns ``x`` itself (bitwise identical, no copy). Otherwise returns float32."""
    if scale == 1.0:
        return x
    if not scale > 0:
        raise ValueError(f"bad resample scale {scale}")
    import torch
    import torch.nn.functional as F

    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected [H, W] or [C, H, W], got {a.shape}")
    t = torch.from_numpy(a)
    t = t[None, None] if a.ndim == 2 else t[None]
    size = resampled_shape(a.shape[-2], a.shape[-1], scale)  # explicit size: scale_factor floors
    out = F.interpolate(t, size=size, mode="bilinear", antialias=True, align_corners=False)
    out = out[0, 0] if a.ndim == 2 else out[0]
    return out.numpy()


# ------------------------------------------------------------------------- train/eval guard
def resample_record(dc, sources: Iterable[str]) -> dict:
    """What gets written next to pixels (shard ``stats.json``, ``embed_info.json``)."""
    return {
        "resample_um_per_px": dc.resample_um_per_px,
        "resample_tol": dc.resample_tol,
        "resample_scales": {s: scale_for(dc, s) for s in sorted(set(sources))},
    }


def check_resample_record(rec: dict, dc, what: str) -> None:
    """Fail loudly if pixels were produced with other resampling settings than ``dc``.

    A record without the keys (written before R7) counts as "not resampled"."""
    have = rec.get("resample_um_per_px")
    want = dc.resample_um_per_px
    same = (have is None and want is None) or (
        have is not None and want is not None and math.isclose(have, want, rel_tol=1e-9)
    )
    if not same:
        raise RuntimeError(
            f"{what}: written with resample_um_per_px={have}, config has {want}; "
            "rebuild the cache / re-run with matching settings, or fix the config"
        )
    if want is None:
        return
    if not math.isclose(float(rec.get("resample_tol", float("nan"))), dc.resample_tol, rel_tol=1e-9):
        raise RuntimeError(f"{what}: resample_tol {rec.get('resample_tol')} != config {dc.resample_tol}")
    for s, sc in (rec.get("resample_scales") or {}).items():
        exp = scale_for(dc, s)
        if not math.isclose(float(sc), exp, rel_tol=1e-9):
            raise RuntimeError(f"{what}: {s} was resampled by {sc}, config gives {exp}")


def check_shards_resample(shards_dir: str | Path, dc, what: str, required: bool = True) -> None:
    """Check a shard directory's ``stats.json`` against ``dc``. A missing file is an error when
    resampling is on and ``required``, otherwise skipped."""
    p = Path(shards_dir) / "stats.json"
    if not p.exists():
        if required and dc.resample_um_per_px is not None:
            raise FileNotFoundError(f"{what}: no {p} to check data.resample_um_per_px against")
        return
    check_resample_record(json.loads(p.read_text(encoding="utf-8")), dc, f"{what} ({p.parent})")

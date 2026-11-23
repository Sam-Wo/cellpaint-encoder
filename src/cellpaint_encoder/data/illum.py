"""Illumination correction: ``corrected = raw / illum`` (CellProfiler convention, DESIGN.md 1.1).

Rule from the design: read dtype and shape from the ``.npy`` header, never assume them (most
sources store float32 at native size; source_7's file size implies float16). Loaded with
``allow_pickle=False``.
"""

from __future__ import annotations

import io

import numpy as np


def read_npy_header(data: bytes) -> tuple[np.dtype, tuple[int, ...], bool]:
    f = io.BytesIO(data)
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return np.dtype(dtype), tuple(shape), bool(fortran)


def load_illum(data: bytes, expect_shape: tuple[int, int] | None = None) -> np.ndarray:
    dtype, shape, _ = read_npy_header(data)
    if dtype.kind != "f":
        raise ValueError(f"illumination function has non-float dtype {dtype}")
    if len(shape) != 2:
        raise ValueError(f"illumination function is not 2-D: {shape}")
    if expect_shape is not None and tuple(shape) != tuple(expect_shape):
        raise ValueError(f"illum shape {shape} != image shape {expect_shape}")
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("illumination function has non-positive or non-finite values")
    return arr


def apply_illum(raw: np.ndarray, illum: np.ndarray) -> np.ndarray:
    if raw.shape != illum.shape:
        raise ValueError(f"image {raw.shape} vs illum {illum.shape}")
    return raw.astype(np.float32) / illum

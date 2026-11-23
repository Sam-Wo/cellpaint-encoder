"""Decode single-page 16-bit greyscale TIFFs (uncompressed or LZW; DESIGN.md 1.1)."""

from __future__ import annotations

import io
import re

import numpy as np
import tifffile

SKIP = re.compile(r"(?:DC_DCAM|_thumb)", re.IGNORECASE)


def is_skipped_file(name: str) -> bool:
    """Camera-calibration (source_10 ``DC_DCAM*``) and thumbnail (source_8 ``_thumb``) files."""
    return bool(SKIP.search(name))


def decode_tiff(data: bytes) -> np.ndarray:
    with tifffile.TiffFile(io.BytesIO(data)) as tf:
        if len(tf.pages) != 1:
            raise ValueError(f"expected a single-page TIFF, got {len(tf.pages)} pages")
        arr = tf.pages[0].asarray()
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D greyscale image, got shape {arr.shape}")
    if arr.dtype not in (np.uint16, np.uint8):
        raise ValueError(f"unexpected dtype {arr.dtype}")
    return arr

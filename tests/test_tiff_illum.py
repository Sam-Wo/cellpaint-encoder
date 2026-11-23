import io

import numpy as np
import pytest
import tifffile

from cellpaint_encoder.data.illum import apply_illum, load_illum, read_npy_header
from cellpaint_encoder.data.tiff_io import decode_tiff, is_skipped_file


def tiff_bytes(arr, compression=None):
    b = io.BytesIO()
    tifffile.imwrite(b, arr, compression=compression)
    return b.getvalue()


@pytest.mark.parametrize("comp", [None, "lzw"])
def test_decode_uint16(comp):
    a = np.random.default_rng(0).integers(0, 65535, (32, 32), dtype=np.uint16)
    out = decode_tiff(tiff_bytes(a, comp))
    assert out.dtype == np.uint16 and np.array_equal(out, a)


def test_decode_rejects_multipage():
    b = io.BytesIO()
    with tifffile.TiffWriter(b) as w:
        w.write(np.zeros((8, 8), np.uint16))
        w.write(np.zeros((8, 8), np.uint16))
    with pytest.raises(ValueError):
        decode_tiff(b.getvalue())


def test_skip_files():
    assert is_skipped_file("x/DC_DCAM_dark.tif") and is_skipped_file("HTS_A01_s1_w1_thumbABC.tif")
    assert not is_skipped_file("1086292082_A01_T0001F001L01A01Z01C01.tif")


def npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_illum_dtype_from_header(dtype):
    f = (np.random.default_rng(1).random((32, 32)) + 0.5).astype(dtype)
    data = npy_bytes(f)
    dt, shape, _ = read_npy_header(data)
    assert dt == np.dtype(dtype) and shape == (32, 32)
    assert len(data) == 32 * 32 * np.dtype(dtype).itemsize + 128  # the size rule in DESIGN 1.1
    out = load_illum(data, expect_shape=(32, 32))
    assert out.dtype == np.float32


def test_illum_divide_and_checks():
    raw = np.full((4, 4), 100, np.uint16)
    illum = np.full((4, 4), 2.0, np.float32)
    assert np.allclose(apply_illum(raw, illum), 50)
    with pytest.raises(ValueError):
        load_illum(npy_bytes(np.zeros((4, 4), np.float32)))  # non-positive
    with pytest.raises(ValueError):
        load_illum(npy_bytes(np.ones((4, 4), np.float32)), expect_shape=(5, 5))
    with pytest.raises(ValueError):
        load_illum(npy_bytes(np.ones((4, 4), np.int32)))


def test_illum_no_pickle():
    b = io.BytesIO()
    np.save(b, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_illum(b.getvalue())

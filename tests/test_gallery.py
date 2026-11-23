"""Gallery bookkeeping without network: concurrent GETs of one key download it once, the byte
cap raises before a download, the mirror is used on re-runs, keys with spaces are URL-encoded."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cellpaint_encoder.data import gallery as gal


class FakeResp:
    def __init__(self, body: bytes):
        self.content, self.status_code, self.headers = body, 200, {"Content-Length": str(len(body))}


def fake_gallery(tmp_path, cap=0):
    g = gal.Gallery(mirror=tmp_path / "mirror", byte_cap=cap)
    calls = {"GET": 0, "HEAD": 0}
    lock = threading.Lock()

    def req(method, url, **kw):
        with lock:
            calls[method] += 1
        time.sleep(0.05)
        return FakeResp(b"x" * 1000)

    g._request = req
    return g, calls


def test_concurrent_gets_download_once(tmp_path):
    g, calls = fake_gallery(tmp_path)
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda _: g.get("cpg0016-jump/a/illum.npy"), range(16)))
    assert all(len(o) == 1000 for o in out)
    assert calls["GET"] == 1 and g.bytes_downloaded == 1000 and g.n_mirror_hits == 15


def test_byte_cap_raises_before_download(tmp_path):
    g, calls = fake_gallery(tmp_path, cap=1500)
    g.get("k1")
    with pytest.raises(gal.ByteCapExceeded):
        g.get("k2")
    assert calls["GET"] == 1


def test_url_encoding_and_key_parsing():
    k = gal.key_from_url(
        "s3://cellpainting-gallery/cpg0016-jump/source_3/images/CP60/images/X__Measurement 1/a.tiff"
    )
    assert k.endswith("Measurement 1/a.tiff")
    assert "Measurement%201" in gal.https_url(k)
    with pytest.raises(ValueError):
        gal.key_from_url("s3://other-bucket/x")

"""Anonymous access to the public ``cellpainting-gallery`` bucket (DESIGN.md 1.1, 1.4).

Plain unsigned HTTPS against the S3 REST endpoint: no AWS account, nothing billable to us.
Every GET goes through :class:`Gallery`, which counts bytes, retries, enforces an optional hard
byte cap, and can keep a local mirror (keys stored under ``mirror/<key>``) so re-runs re-use
files instead of re-downloading them.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BUCKET = "cellpainting-gallery"
ENDPOINT = f"https://{BUCKET}.s3.amazonaws.com"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class ByteCapExceeded(RuntimeError):
    pass


def key_from_url(url: str) -> str:
    """``s3://cellpainting-gallery/cpg0016-jump/...`` or an https URL -> bucket key."""
    if url.startswith("s3://"):
        b, _, key = url[5:].partition("/")
        if b != BUCKET:
            raise ValueError(f"unexpected bucket {b!r}")
        return key
    if url.startswith(ENDPOINT + "/"):
        return urllib.parse.unquote(url[len(ENDPOINT) + 1 :])
    return url.lstrip("/")


def https_url(key: str) -> str:
    # source_3 measurement directories contain spaces ("Measurement 1")
    return f"{ENDPOINT}/{urllib.parse.quote(key, safe='/')}"


class Gallery:
    def __init__(self, mirror: str | Path | None = None, byte_cap: float = 0, retries: int = 5,
                 timeout: float = 120, ledger: str | Path | None = None):
        self.mirror = Path(mirror) if mirror else None
        self.byte_cap = float(byte_cap or 0)
        self.retries = retries
        self.timeout = timeout
        self.bytes_downloaded = 0
        self.n_get = 0
        self.n_mirror_hits = 0
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._local = threading.local()
        self.ledger = Path(ledger) if ledger else (self.mirror / "_download_ledger.json" if self.mirror else None)
        self._ledger_total = self._read_ledger()

    # ------------------------------------------------------------------ bookkeeping
    def _read_ledger(self) -> float:
        if self.ledger and self.ledger.exists():
            return float(json.loads(self.ledger.read_text(encoding="utf-8")).get("bytes", 0))
        return 0.0

    def _write_ledger(self) -> None:
        if not self.ledger:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps({"bytes": self._ledger_total + self.bytes_downloaded}), encoding="utf-8")
        os.replace(tmp, self.ledger)

    @property
    def total_bytes_all_runs(self) -> float:
        return self._ledger_total + self.bytes_downloaded

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
        return s

    def _request(self, method: str, url: str, **kw) -> requests.Response:
        last = None
        for k in range(self.retries):
            try:
                r = self._session().request(method, url, timeout=self.timeout, **kw)
                if r.status_code in (200, 206):
                    return r
                if r.status_code in (403, 404):
                    raise FileNotFoundError(f"HTTP {r.status_code}: {url}")
                last = RuntimeError(f"HTTP {r.status_code}: {url}")
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
            time.sleep(min(2**k, 30))
        raise RuntimeError(f"failed after {self.retries} tries: {url}") from last

    # ------------------------------------------------------------------ API
    def mirror_path(self, key: str) -> Path | None:
        return self.mirror / key if self.mirror else None

    def head_size(self, key: str) -> int:
        r = self._request("HEAD", https_url(key))
        return int(r.headers["Content-Length"])

    def get(self, key: str, byte_range: tuple[int, int] | None = None) -> bytes:
        """GET one object. Concurrent requests for the same key are serialised, so a key is
        downloaded once even when many threads ask for it (e.g. a plate's illum functions)."""
        key = key_from_url(key)
        if byte_range is not None:
            return self._get(key, byte_range)
        with self._lock:
            kl = self._key_locks.setdefault(key, threading.Lock())
        with kl:
            return self._get(key, None)

    def _get(self, key: str, byte_range: tuple[int, int] | None) -> bytes:
        mp = self.mirror_path(key)
        if byte_range is None and mp is not None and mp.exists():
            with self._lock:
                self.n_mirror_hits += 1
            return mp.read_bytes()
        if self.byte_cap:
            need = self.head_size(key) if byte_range is None else byte_range[1] - byte_range[0] + 1
            with self._lock:
                if self.total_bytes_all_runs + need > self.byte_cap:
                    raise ByteCapExceeded(
                        f"GET {key} ({need} B) would exceed byte cap {self.byte_cap:.0f} "
                        f"(already {self.total_bytes_all_runs:.0f})")
        headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
        data = self._request("GET", https_url(key), headers=headers).content
        with self._lock:
            self.bytes_downloaded += len(data)
            self.n_get += 1
            self._write_ledger()
        if byte_range is None and mp is not None:
            mp.parent.mkdir(parents=True, exist_ok=True)
            tmp = mp.with_name(mp.name + f".tmp{threading.get_ident()}")
            tmp.write_bytes(data)
            os.replace(tmp, mp)
        return data

    def list(self, prefix: str, delimiter: str | None = None, max_keys: int | None = None):
        """Yield ``(key, size)`` for objects and ``(prefix, None)`` for common prefixes."""
        token, n = None, 0
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            if max_keys:
                q["max-keys"] = str(max_keys)
            r = self._request("GET", ENDPOINT + "/?" + urllib.parse.urlencode(q))
            with self._lock:
                self.bytes_downloaded += len(r.content)
            root = ET.fromstring(r.content)
            for c in root.findall(f"{_NS}Contents"):
                yield c.find(f"{_NS}Key").text, int(c.find(f"{_NS}Size").text)
                n += 1
            for c in root.findall(f"{_NS}CommonPrefixes"):
                yield c.find(f"{_NS}Prefix").text, None
            if max_keys and n >= max_keys:
                return
            if root.find(f"{_NS}IsTruncated").text != "true":
                return
            token = root.find(f"{_NS}NextContinuationToken").text


def load_data_key(source: str, batch: str, plate: str, with_illum: bool = False) -> str:
    name = "load_data_with_illum.csv" if with_illum else "load_data.csv"
    return f"cpg0016-jump/{source}/workspace/load_data_csv/{batch}/{plate}/{name}"


def illum_key(source: str, batch: str, plate: str, channel: str) -> str:
    """Illumination functions are named by plate barcode (also for Phenix sources, 1.1)."""
    return f"cpg0016-jump/{source}/images/{batch}/illum/{plate}/{plate}_Illum{channel}.npy"


def cp_profile_key(source: str, batch: str, plate: str) -> str:
    return f"cpg0016-jump/{source}/workspace/profiles/{batch}/{plate}/{plate}.parquet"

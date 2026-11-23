"""WebDataset-format tar shards: ``<key>.npy.zst`` (zstd-compressed ``.npy`` of uint8 ``[5,H,W]``)
plus ``<key>.json`` metadata (DESIGN.md 1.3).

The shards are plain POSIX (ustar) tars, so the ``webdataset`` library or ``tar`` can read them.
Training does **not** stream them through a shuffle buffer: an ``index.parquet`` records the byte
offset of every sample, and the sampler picks samples by index. That makes the sample order a pure
function of ``(seed, epoch, step)`` and resume exact, with no hidden buffer state (see
``sampler.py``). Shards should live on local NVMe (RUNBOOK.md, "storage").
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard

HEADER = 512  # ustar header size; names are kept short so there are no extension headers


def encode_array(arr: np.ndarray, level: int = 3) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return zstandard.ZstdCompressor(level=level).compress(buf.getvalue())


def decode_array(data: bytes) -> np.ndarray:
    raw = zstandard.ZstdDecompressor().decompress(data)
    return np.load(io.BytesIO(raw), allow_pickle=False)


class ShardWriter:
    def __init__(self, out_dir: str | Path, prefix: str = "shard", max_bytes: float = 1e9, zstd_level: int = 3):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.max_bytes, self.level = prefix, max_bytes, zstd_level
        self.n_shard = -1
        self.tf: tarfile.TarFile | None = None
        self.rows: list[dict] = []
        self.raw_bytes = 0
        self.zst_bytes = 0
        self._sum = None
        self._sumsq = None
        self._npix = 0

    def _open_next(self):
        if self.tf is not None:
            self.tf.close()
        self.n_shard += 1
        self.name = f"{self.prefix}-{self.n_shard:06d}.tar"
        self.tf = tarfile.open(self.dir / self.name, "w", format=tarfile.USTAR_FORMAT)

    def _add(self, name: str, data: bytes) -> int:
        if len(name) > 99:
            raise ValueError(f"member name too long for ustar: {name}")
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = 0  # deterministic bytes
        pos = self.tf.fileobj.tell()
        self.tf.addfile(ti, io.BytesIO(data))
        return pos + HEADER

    def write(self, key: str, arr: np.ndarray, meta: dict) -> None:
        if arr.dtype != np.uint8:
            raise TypeError("shards hold uint8 arrays")
        if self.tf is None or self.tf.fileobj.tell() >= self.max_bytes:
            self._open_next()
        data = encode_array(arr, self.level)
        off = self._add(f"{key}.npy.zst", data)
        self._add(f"{key}.json", json.dumps(meta, sort_keys=True).encode())
        self.rows.append({"key": key, "shard": self.name, "offset": off, "size": len(data),
                          "shape": list(arr.shape), **meta})
        self.raw_bytes += arr.nbytes
        self.zst_bytes += len(data)
        x = arr.reshape(arr.shape[0], -1).astype(np.float64) / 255.0
        s, ss = x.sum(1), (x * x).sum(1)
        self._sum = s if self._sum is None else self._sum + s
        self._sumsq = ss if self._sumsq is None else self._sumsq + ss
        self._npix += x.shape[1]

    def close(self, extra_stats: dict | None = None) -> pd.DataFrame:
        if self.tf is not None:
            self.tf.close()
            self.tf = None
        idx = pd.DataFrame(self.rows)
        idx.to_parquet(self.dir / "index.parquet", index=False)
        stats = {"n_samples": len(self.rows), "n_shards": self.n_shard + 1, "raw_bytes": self.raw_bytes,
                 "zstd_bytes": self.zst_bytes,
                 "zstd_ratio": (self.zst_bytes / self.raw_bytes) if self.raw_bytes else None}
        if self._npix:
            mean = self._sum / self._npix
            std = np.sqrt(np.maximum(self._sumsq / self._npix - mean**2, 1e-12))
            stats["channel_mean"] = mean.round(6).tolist()
            stats["channel_std"] = std.round(6).tolist()
        stats.update(extra_stats or {})
        (self.dir / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        return idx


def load_index(shards_dir: str | Path) -> pd.DataFrame:
    p = Path(shards_dir) / "index.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no shard index at {p}; run scripts/write_shards.py first")
    return pd.read_parquet(p)


def load_stats(shards_dir: str | Path) -> dict:
    return json.loads((Path(shards_dir) / "stats.json").read_text(encoding="utf-8"))


def iter_shard(path: str | Path):
    """Sequential WebDataset-style iteration: yields (key, array, meta)."""
    with tarfile.open(path, "r") as tf:
        pending: dict[str, dict] = {}
        for m in tf:
            key, _, ext = m.name.partition(".")
            data = tf.extractfile(m).read()
            d = pending.setdefault(key, {})
            d[ext] = data
            if "npy.zst" in d and "json" in d:
                yield key, decode_array(d["npy.zst"]), json.loads(d["json"])
                del pending[key]


class ShardReader:
    """Random access by (shard, offset, size). Keeps file handles open per process (and thread),
    with an optional LRU cache of decoded arrays (``data.tile_cache_mb``)."""

    def __init__(self, shards_dir: str | Path, cache_mb: int = 0):
        self.dir = Path(shards_dir)
        self.cache_bytes = int(cache_mb) * 2**20
        self._local = threading.local()
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._cached = 0
        self.hits = 0

    def __getstate__(self):  # picklable for DataLoader workers (spawn on Windows)
        return {"dir": self.dir, "cache_bytes": self.cache_bytes}

    def __setstate__(self, st):
        self.__init__(st["dir"], st["cache_bytes"] // 2**20)

    def _fh(self, shard: str):
        fhs = getattr(self._local, "fhs", None)
        if fhs is None:
            fhs = self._local.fhs = {}
        if shard not in fhs:
            fhs[shard] = open(self.dir / shard, "rb", buffering=0)
        return fhs[shard]

    def read(self, shard: str, offset: int, size: int) -> np.ndarray:
        k = (shard, int(offset))
        if self.cache_bytes and k in self._cache:
            self._cache.move_to_end(k)
            self.hits += 1
            return self._cache[k]
        fh = self._fh(shard)
        data = os.pread(fh.fileno(), size, offset) if hasattr(os, "pread") else _seek_read(fh, offset, size)
        if len(data) != size:
            raise OSError(f"short read from {shard} at {offset}")
        arr = decode_array(data)
        if self.cache_bytes:
            self._cache[k] = arr
            self._cached += arr.nbytes
            while self._cached > self.cache_bytes and self._cache:
                _, old = self._cache.popitem(last=False)
                self._cached -= old.nbytes
        return arr


def _seek_read(fh, offset: int, size: int) -> bytes:
    fh.seek(offset)
    return fh.read(size)


class TileDataset:
    """Map-style dataset over a shard index; returns uint8 tensors ``[5, H, W]``. Picklable."""

    def __init__(self, index: pd.DataFrame, shards_dir: str | Path, cache_mb: int = 0):
        self.shard = index["shard"].to_numpy().astype(str)
        self.offset = index["offset"].to_numpy().astype(np.int64)
        self.size = index["size"].to_numpy().astype(np.int64)
        self.reader = ShardReader(shards_dir, cache_mb)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, i):
        import torch

        arr = self.reader.read(self.shard[i], int(self.offset[i]), int(self.size[i]))
        return torch.from_numpy(arr)


def collate_tiles(batch):
    import torch

    return torch.stack(batch)

"""Source-balanced batch sampler with a DMSO cap (DESIGN.md 3.4), stateless and exactly resumable.

Micro-batch ``m`` is a pure function of ``(seed, m)``: with ``P`` micro-batches per epoch,
``epoch = m // P`` and ``b = m % P``. Each source contributes ``k = micro_batch / n_sources``
samples (remainder spread over the first sources), of which a fixed ``n_dmso = min(floor(cap*k),
round(k * dmso_fraction))`` come from that source's DMSO pool. Within an epoch each pool is read
in the order of a seeded permutation; a pool that is exhausted (a small source such as source_10,
oversampled ~2x) continues with a fresh permutation (cycle ``c``), seeded by ``(seed, epoch,
source, pool, c)``. Resuming at micro-batch ``m`` therefore needs only ``m``; there is no shuffle
buffer whose state could run ahead of the checkpoint.

``balance="natural"`` (ablation A1) treats all sources as one pool.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little")


class BalancedBatchSampler:
    def __init__(self, index: pd.DataFrame, micro_batch: int, seed: int, micro_per_epoch: int,
                 sources: list[str] | tuple[str, ...] | None = None, dmso_cap: float = 0.2,
                 balance: str = "source", start: int = 0, stop: int | None = None):
        if balance not in ("source", "natural"):
            raise ValueError(f"balance must be source|natural, got {balance!r}")
        self.micro_batch, self.seed, self.P = int(micro_batch), int(seed), int(micro_per_epoch)
        self.start, self.stop = int(start), stop
        src = index["source"].to_numpy() if balance == "source" else np.full(len(index), "all")
        names = sorted(set(src)) if sources is None or balance == "natural" else list(sources)
        missing = [s for s in names if s not in set(src)]
        if missing:
            raise ValueError(f"no samples for sources {missing}")
        is_dmso = index["is_dmso"].to_numpy().astype(bool)
        self.pools: list[tuple[np.ndarray, np.ndarray]] = []
        for s in names:
            sel = src == s
            self.pools.append((np.flatnonzero(sel & ~is_dmso), np.flatnonzero(sel & is_dmso)))
        self.sources = names
        n = len(names)
        base, rem = divmod(self.micro_batch, n)
        self.k = [base + (1 if i < rem else 0) for i in range(n)]
        self.n_dmso = []
        for (nd, d), k in zip(self.pools, self.k):
            frac = len(d) / max(len(d) + len(nd), 1)
            nd_ = min(int(np.floor(dmso_cap * k)), int(round(k * frac)))
            if len(nd) == 0:  # only DMSO available for this source
                nd_ = k
            self.n_dmso.append(nd_ if len(d) else 0)

    def __len__(self):
        return (self.stop - self.start) if self.stop is not None else 2**62

    @lru_cache(maxsize=64)  # noqa: B019 - small, per-instance pools
    def _perm(self, epoch: int, s: int, pool: int, cycle: int) -> np.ndarray:
        idx = self.pools[s][pool]
        return idx[np.random.default_rng(_seed(self.seed, epoch, s, pool, cycle)).permutation(len(idx))]

    def _take(self, epoch: int, s: int, pool: int, pos: int, n: int) -> np.ndarray:
        size = len(self.pools[s][pool])
        out = []
        while n > 0:
            cyc, off = divmod(pos, size)
            chunk = self._perm(epoch, s, pool, cyc)[off : off + n]
            out.append(chunk)
            pos += len(chunk)
            n -= len(chunk)
        return np.concatenate(out) if out else np.empty(0, np.int64)

    def batch(self, m: int) -> list[int]:
        epoch, b = divmod(int(m), self.P)
        out = []
        for s, (k, nd) in enumerate(zip(self.k, self.n_dmso)):
            n_non = k - nd
            if n_non:
                out.append(self._take(epoch, s, 0, b * n_non, n_non))
            if nd:
                out.append(self._take(epoch, s, 1, b * nd, nd))
        idx = np.concatenate(out)
        # shuffle within the batch so sources are not in blocks (does not change the set)
        idx = idx[np.random.default_rng(_seed(self.seed, "order", m)).permutation(len(idx))]
        return idx.tolist()

    def __iter__(self):
        m = self.start
        while self.stop is None or m < self.stop:
            yield self.batch(m)
            m += 1

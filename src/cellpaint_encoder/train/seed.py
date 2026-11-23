"""Seeding and RNG-state capture (DESIGN.md 3.3).

One ``seed`` drives Python, numpy and torch. Loader workers get
``seed + 1000*rank + worker_id + 10**4*epoch``; they draw no sample-affecting randomness (crop
positions and augmentations are drawn on the device from a generator seeded per micro-batch), so
worker seeding cannot break exact resume.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def micro_batch_seed(seed: int, m: int) -> int:
    h = hashlib.sha256(f"aug|{seed}|{m}".encode()).digest()
    return int.from_bytes(h[:8], "little") & (2**63 - 1)


def worker_seed(seed: int, worker_id: int, epoch: int = 0, rank: int = 0) -> int:
    return seed + 1000 * rank + worker_id + 10**4 * epoch


def make_worker_init(seed: int, epoch: int = 0, rank: int = 0):
    return _WorkerInit(seed, epoch, rank)


class _WorkerInit:  # picklable (spawn on Windows)
    def __init__(self, seed, epoch, rank):
        self.seed, self.epoch, self.rank = seed, epoch, rank

    def __call__(self, worker_id):
        s = worker_seed(self.seed, worker_id, self.epoch, self.rank)
        random.seed(s)
        np.random.seed(s % 2**32)
        torch.manual_seed(s)


def get_rng_state() -> dict:
    py = random.getstate()
    npst = np.random.get_state()
    st = {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": {
            "name": npst[0],
            "keys": torch.from_numpy(npst[1].astype(np.int64)),
            "pos": int(npst[2]),
            "has_gauss": int(npst[3]),
            "cached": float(npst[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    py = st["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = st["numpy"]
    np.random.set_state(
        (n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"])
    )
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])

"""Benchmarks for the G0/G1 gates (DESIGN.md 3.4, 5.4). Run on the pod before any GPU spend.

# S3 read throughput (G0: >= 150 MB/s sustained over 10 min); reads sites listed in the manifest
python scripts/bench_io.py s3 --config configs/full.yaml --seconds 600 --workers 64
# cache-build CPU rate: decode + illum + plate norm + tiles (G0: >= 40 sites/s on 8 vCPU)
python scripts/bench_io.py cache --config configs/full.yaml --sites 400 --workers 8
# training-loader rate from the shards alone (needs > 280 samples/s with the planned workers)
python scripts/bench_io.py loader --config configs/full.yaml --batches 50
# training step rate with synthetic tiles (no data), any device; reports FLOPs/s
python scripts/bench_io.py step --config configs/full.yaml --steps 20 [optim.batch_size=...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from cellpaint_encoder import CHANNELS
from cellpaint_encoder.config import load_config


def vit_macs(n_tokens: int, dim: int = 384, depth: int = 12, in_chans: int = 5, patch: int = 16) -> float:
    per_block = 12 * n_tokens * dim**2 + 2 * n_tokens**2 * dim
    return depth * per_block + (n_tokens - 1) * in_chans * patch**2 * dim


def dino_flops_per_sample(cfg) -> float:
    d, m = cfg.dino, cfg.model
    tok = lambda s: (s // m.patch_size) ** 2 + 1  # noqa: E731
    head = (
        m.embed_dim * d.hidden_dim
        + d.hidden_dim**2
        + d.hidden_dim * d.bottleneck_dim
        + d.bottleneck_dim * d.out_dim
    )
    g = vit_macs(tok(d.global_crop_size), m.embed_dim, m.depth, m.in_chans, m.patch_size) + head
    loc = vit_macs(tok(d.local_crop_size), m.embed_dim, m.depth, m.in_chans, m.patch_size) + head
    student_fwd = 2 * g + d.n_local_crops * loc
    teacher_fwd = 2 * g
    return 2 * (3 * student_fwd + teacher_fwd)  # FLOPs = 2 x MACs; backward ~ 2 x forward


def bench_step(cfg, steps: int) -> dict:
    from cellpaint_encoder.train.loop import Trainer

    tmp = Path("runs/_bench_step")
    shards = tmp / "shards"
    if not (shards / "index.parquet").exists():
        from cellpaint_encoder.data.shards import ShardWriter

        w = ShardWriter(shards, "train", 1e9, 1)
        rng = np.random.default_rng(0)
        for s in cfg.data.train_sources:
            for i in range(max(cfg.optim.batch_size, 8)):
                meta = {"source": s, "is_dmso": False, "plate": "p", "well": "A01", "site": 1}
                w.write(
                    f"{s}_{i}",
                    rng.integers(0, 255, (5, cfg.data.tile_size, cfg.data.tile_size), dtype=np.uint8),
                    meta,
                )
        w.close()
    from cellpaint_encoder.config import load_config as lc

    over = [
        f"data.shards_dir={shards.as_posix()}",
        "data.channel_mean=[0.1,0.1,0.1,0.1,0.1]",
        "data.channel_std=[0.15,0.15,0.15,0.15,0.15]",
        f"run.max_steps={steps}",
        "run.ckpt_every_minutes=0",
        "log.tensorboard=false",
        f"run.log_every_steps={steps}",
    ]
    c2 = lc(None, [*_flatten_overrides(cfg), *over])
    tr = Trainer(c2, tmp / "run", install_signals=False)
    t0 = time.time()
    tr.train(resume=None)
    dt = time.time() - t0
    log = pd.read_csv(tmp / "run" / "train_log.csv")
    sps = float(log.samples_per_s.iloc[-1])
    fl = dino_flops_per_sample(c2)
    return {
        "device": str(tr.device),
        "batch": c2.optim.batch_size,
        "steps": steps,
        "wall_s": dt,
        "samples_per_s_steady": sps,
        "flops_per_sample": fl,
        "achieved_tflops": sps * fl / 1e12,
        "threads": torch.get_num_threads(),
    }


def _flatten_overrides(cfg) -> list[str]:
    from cellpaint_encoder.config import to_dict

    out = []

    def rec(d, p=""):
        for k, v in d.items():
            if isinstance(v, dict):
                rec(v, f"{p}{k}.")
            else:
                out.append(f"{p}{k}={json.dumps(v)}")

    rec(to_dict(cfg))
    return out


def bench_loader(cfg, batches: int, workers: int | None) -> dict:
    from cellpaint_encoder.data.sampler import BalancedBatchSampler
    from cellpaint_encoder.data.shards import TileDataset, collate_tiles, load_index

    idx = load_index(cfg.data.shards_dir)
    nw = cfg.data.num_workers if workers is None else workers
    micro = cfg.optim.batch_size // cfg.optim.accum_steps
    s = BalancedBatchSampler(idx, micro, 0, 10**6, sources=list(cfg.data.train_sources), stop=batches + 2)
    dl = torch.utils.data.DataLoader(
        TileDataset(idx, cfg.data.shards_dir),
        batch_sampler=s,
        num_workers=nw,
        collate_fn=collate_tiles,
        persistent_workers=False,
        prefetch_factor=cfg.data.prefetch_factor if nw else None,
    )
    it = iter(dl)
    next(it), next(it)  # warm-up
    t0, n = time.time(), 0
    for b in it:
        n += b.shape[0]
    dt = time.time() - t0
    mb = idx["size"].mean() / 1e6
    return {
        "workers": nw,
        "samples": n,
        "samples_per_s": n / dt,
        "compressed_MB_per_s": n * mb / dt,
        "samples_per_s_per_worker": n / dt / max(nw, 1),
        "mean_compressed_MB": mb,
    }


_BW: dict = {}


def _bench_init(cfg_path, overrides):
    from cellpaint_encoder.data.gallery import Gallery
    from cellpaint_encoder.data.sites import IllumCache

    cfg = load_config(cfg_path, overrides)
    g = Gallery(mirror=cfg.data.raw_mirror, byte_cap=cfg.data.byte_cap, ledger=False)
    if os.environ.get("BENCH_IN_MEMORY"):  # exclude local file reads (e.g. antivirus scanning on Windows)
        mem, get = {}, g.get

        def cached(key, byte_range=None):
            if key not in mem:
                mem[key] = get(key, byte_range)
            return mem[key]

        g.get = cached
    if cfg.data.resample_um_per_px is not None:  # as in write_shards' workers
        import torch

        torch.set_num_threads(1)
    _BW.update(cfg=cfg, g=g, il=IllumCache(g, max_plates=4))


def _bench_site(r: dict) -> int:
    from cellpaint_encoder.data.manifest import _rng
    from cellpaint_encoder.data.resample import scale_for
    from cellpaint_encoder.data.sites import read_site_uint8
    from cellpaint_encoder.data.tiles import training_tiles

    dc = _BW["cfg"].data
    scale = scale_for(dc, r["source"])  # R7 resample included, as in write_shards
    u8 = read_site_uint8(_BW["g"], _BW["il"], pd.Series(r), [0.0] * 5, [1000.0] * 5, scale=scale)
    return len(
        training_tiles(u8, dc.tile_size, dc.tiles_per_site, dc.fg_min, _rng(0, r["plate"], r["well"]))[0]
    )


def bench_cache(cfg_path, overrides, cfg, n_sites: int, procs: int) -> dict:
    """decode + illum + normalise + tile, one site per task, ``procs`` worker processes."""
    from concurrent.futures import ProcessPoolExecutor

    m = pd.read_parquet(cfg.data.manifest)
    rows = m[m.train | (m.eval_set != "")].head(n_sites).to_dict("records")
    work = (rows * max(1, (40 * procs) // max(len(rows), 1)))[: max(len(rows), 40 * procs)]
    with ProcessPoolExecutor(procs, initializer=_bench_init, initargs=(cfg_path, overrides)) as ex:
        list(ex.map(_bench_site, rows[: procs * 2] or rows))  # warm-up: illum + page cache
        t0 = time.time()
        list(ex.map(_bench_site, work, chunksize=2))
        dt = time.time() - t0
    return {
        "sites": len(work),
        "procs": procs,
        "sites_per_s": len(work) / dt,
        "note": "CPU only when data.raw_mirror holds the files (no network)",
    }


def bench_s3(cfg, seconds: float, workers: int) -> dict:
    from cellpaint_encoder.data.gallery import Gallery

    m = pd.read_parquet(cfg.data.manifest)
    keys = [k for c in CHANNELS for k in m[m.train][f"key_{c}"]]
    rng = np.random.default_rng(0)
    keys = list(rng.permutation(keys))
    g = Gallery(mirror=None, byte_cap=cfg.data.byte_cap)
    t0 = time.time()
    done = []

    def one(k):
        if time.time() - t0 > seconds:
            return 0
        return len(g.get(k))

    with ThreadPoolExecutor(workers) as ex:
        for b in ex.map(one, keys):
            done.append(b)
            if time.time() - t0 > seconds:
                break
    dt = time.time() - t0
    return {
        "objects": int(sum(1 for b in done if b)),
        "bytes": int(sum(done)),
        "seconds": dt,
        "MB_per_s": sum(done) / dt / 1e6,
        "workers": workers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["s3", "cache", "loader", "step"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--sites", type=int, default=400)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_intermixed_args()
    cfg = load_config(a.config, a.overrides)
    if cfg.run.threads:
        torch.set_num_threads(cfg.run.threads)
    if a.mode == "step":
        out = bench_step(cfg, a.steps)
    elif a.mode == "loader":
        out = bench_loader(cfg, a.batches, a.workers)
    elif a.mode == "cache":
        out = bench_cache(a.config, a.overrides, cfg, a.sites, a.workers or 8)
    else:
        out = bench_s3(cfg, a.seconds, a.workers or 64)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

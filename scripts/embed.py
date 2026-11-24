"""Embed evaluation sites with several encoders in one decode pass (DESIGN.md 1.3 "embed", 4.6).

    py -3.13 -I -X utf8 scripts/embed.py --config configs/full.yaml --sets E1 \
        --encoder ours=runs/full-<hash>/teacher_backbone.pt --encoder imagenet_dino --encoder random_init \
        --out runs/embeddings/final

Encoder specs: ``ours=<teacher_backbone.pt or checkpoint>``, ``imagenet_dino``, ``random_init``,
``deepcellmorpho=<converted .pt>[:arch]`` and ``openphenom=<repo_dir>`` (the last two only inside the
sandbox, see RUNBOOK.md). Our model, ImageNet and random use the training shards' channel
statistics. Embedding source_6 is allowed; metrics on it stay locked (evaluate.py).

Pixel size (R7): every encoder sees sites resampled to ``data.resample_um_per_px`` (sources within
``data.resample_tol`` stay native). The script fails if the training shards' ``stats.json`` or the
``ours=`` checkpoint's config were built with other resampling settings, and records the applied
per-source scales in each ``embed_info.json``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json

import pandas as pd
import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.gallery import Gallery
from cellpaint_encoder.data.plate_norm import compute_plate_norms
from cellpaint_encoder.data.resample import check_resample_record, check_shards_resample, resample_record
from cellpaint_encoder.data.shards import load_stats
from cellpaint_encoder.data.sites import IllumCache
from cellpaint_encoder.embed.run import embed_manifest
from cellpaint_encoder.train.loop import resolve_device


def build_encoder(spec: str, cfg, device, window):
    name, _, arg = spec.partition("=")
    kw = dict(crop=cfg.eval.crop, grid=cfg.eval.grid, window=window)
    amp = device.type == "cuda"
    if name == "ours":
        from cellpaint_encoder.embed.baselines import ours

        enc = ours.build(arg, device, amp=amp, **kw)
        # the checkpoint's training cache must have been resampled like the pixels we feed it
        check_resample_record(
            resample_record(enc.train_cfg.data, enc.train_cfg.data.train_sources),
            cfg.data,
            f"checkpoint {arg}",
        )
        return enc
    st = load_stats(cfg.data.shards_dir)
    check_resample_record(st, cfg.data, f"training shards {cfg.data.shards_dir} (channel statistics)")
    mean, std = st["channel_mean"], st["channel_std"]
    if name == "imagenet_dino":
        from cellpaint_encoder.embed.baselines import imagenet_dino

        return imagenet_dino.build(cfg, mean, std, device, amp=amp, **kw)
    if name == "random_init":
        from cellpaint_encoder.embed.baselines import random_init

        return random_init.build(cfg, mean, std, device, seed=cfg.run.seed, amp=amp, **kw)
    if name == "deepcellmorpho":
        from cellpaint_encoder.embed.baselines import deepcellmorpho

        path, _, arch = arg.partition(":")
        return deepcellmorpho.build(path, arch or "dinov2_vitb16", device, window=window)
    if name == "openphenom":
        from cellpaint_encoder.embed.baselines import openphenom

        return openphenom.build(arg, device=device, window=window)
    raise ValueError(f"unknown encoder {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", action="append", required=True)
    ap.add_argument("--sets", default="E1", help="comma-separated eval sets (E1,E2,E3,E4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=None, help="common central window (768 sensitivity run)")
    ap.add_argument("--max-sites", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    if cfg.run.threads:
        torch.set_num_threads(cfg.run.threads)
    device = resolve_device(cfg.run.device)
    m = pd.read_parquet(cfg.data.manifest)
    rows = m[m.eval_set.isin(a.sets.split(","))].reset_index(drop=True)
    if a.max_sites:
        rows = rows.iloc[: a.max_sites]
    resample_info = resample_record(cfg.data, set(rows.source))  # raises for a source without a pixel size
    check_shards_resample(cfg.data.shards_dir, cfg.data, "training shards", required=False)
    g = Gallery(mirror=cfg.data.raw_mirror, byte_cap=cfg.data.byte_cap)
    norms = compute_plate_norms(
        m,
        g,
        IllumCache(g),
        Path(cfg.data.manifest).with_name("plate_norm.json"),
        cfg.data.pct_lo,
        cfg.data.pct_hi,
        cfg.data.manifest_seed,
        a.workers,
        plates=set(rows.plate),
    )
    encs = [build_encoder(s, cfg, device, a.window) for s in a.encoder]
    info = embed_manifest(
        rows, encs, g, norms, Path(a.out), cfg.eval.batch_size, a.workers, resample_info=resample_info
    )
    info["sites_per_s"] = info["sites"] / max(info["seconds"], 1e-9)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

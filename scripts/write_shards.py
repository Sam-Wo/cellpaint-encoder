"""Build the tile cache (training shards) and the dev cache in one pass (DESIGN.md 1.3).

    py -3.13 -I -X utf8 scripts/write_shards.py --config configs/full.yaml [--only train|dev|norm]

1. per-plate DMSO percentiles from the manifest's ``norm`` sites -> ``plate_norm.json``
2. training tiles: illum-corrected, plate-normalised uint8, 2 x 448^2 per site with the Otsu
   DNA foreground filter -> ``data.shards_dir``
3. dev cache: anchor wells + 4 DMSO wells per plate of the dev sources, centred 896^2 sites
   -> ``data.dev_shards_dir``

Everything is deterministic given the manifest, the code and ``data.manifest_seed``. Timing
(fetch vs CPU processing), the zstd ratio and the foreground drop rate go into ``stats.json``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.gallery import Gallery
from cellpaint_encoder.data.manifest import _rng
from cellpaint_encoder.data.normalize import normalise_site
from cellpaint_encoder.data.plate_norm import compute_plate_norms
from cellpaint_encoder.data.shards import ShardWriter
from cellpaint_encoder.data.sites import IllumCache, read_site
from cellpaint_encoder.data.tiles import centre_crop, training_tiles

META_COLS = ["source", "batch", "plate", "well", "site", "row", "col", "compound_id", "is_dmso"]


class Timer:
    def __init__(self):
        self.t = {}
        self._lock = threading.Lock()

    def add(self, k, dt):
        with self._lock:
            self.t[k] = self.t.get(k, 0.0) + dt


def site_meta(r) -> dict:
    d = {c: getattr(r, c) for c in META_COLS}
    d["site"], d["row"], d["col"] = int(d["site"]), int(d["row"]), int(d["col"])
    d["is_dmso"] = bool(d["is_dmso"])
    return d


def run_pass(rows: pd.DataFrame, fn, workers: int, chunk: int = 256):
    """Map ``fn`` over rows with a bounded thread pool, yielding results in row order."""
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(rows), chunk):
            yield from ex.map(fn, list(rows.iloc[i : i + chunk].itertuples()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", choices=["norm", "train", "dev"], default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dc = cfg.data
    m = pd.read_parquet(dc.manifest)
    g = Gallery(mirror=dc.raw_mirror, byte_cap=dc.byte_cap)
    il = IllumCache(g)
    timer = Timer()
    norm_path = Path(dc.manifest).with_name("plate_norm.json")
    t_all = time.time()
    t0 = time.time()
    norms = compute_plate_norms(m, g, il, norm_path, dc.pct_lo, dc.pct_hi, dc.manifest_seed, dc.s3_workers)
    timer.add("norm", time.time() - t0)
    need = set(m[m.train | m.dev].plate) - set(norms)
    if need and dc.norm_mode == "per_plate_dmso":
        raise RuntimeError(f"plates without normalisation sites: {sorted(need)[:5]}")
    if a.only == "norm":
        return

    def process_train(r):
        t0 = time.time()
        site = read_site(g, il, r)
        t1 = time.time()
        u8 = normalise_site(site, dc.norm_mode, norms.get(r.plate), dc.pct_lo, dc.pct_hi)
        rng = _rng(dc.manifest_seed, "tiles", r.plate, r.well, str(r.site))
        tiles, info = training_tiles(u8, dc.tile_size, dc.tiles_per_site, dc.fg_min, rng)
        timer.add("read", t1 - t0)
        timer.add("process", time.time() - t1)
        return r, tiles, info

    def process_dev(r):
        t0 = time.time()
        site = read_site(g, il, r)
        t1 = time.time()
        u8 = normalise_site(site, "per_plate_dmso", norms.get(r.plate), dc.pct_lo, dc.pct_hi)
        u8 = np.ascontiguousarray(centre_crop(u8, dc.dev_site_size))
        timer.add("read", t1 - t0)
        timer.add("process", time.time() - t1)
        return r, u8

    summary = {"plate_norm_plates": len(norms)}
    if a.only in (None, "train"):
        tr = m[m.train].reset_index(drop=True)
        tr = tr.iloc[_rng(dc.manifest_seed, "shard-order").permutation(len(tr))]  # mix plates in shards
        w = ShardWriter(dc.shards_dir, "train", dc.shard_max_bytes, dc.zstd_level)
        t0, n_rej, n_cand, fg = time.time(), 0, 0, []
        for r, tiles, info in run_pass(tr, process_train, dc.s3_workers):
            n_rej += info["rejected"]
            n_cand += len(info["fg_fracs"])
            fg += info["fg_fracs"]
            for k, t in enumerate(tiles):
                w.write(
                    f"{r.source}__{r.plate}__{r.well}__s{int(r.site)}__t{k}", t, {**site_meta(r), "tile": k}
                )
        dt = time.time() - t0
        st = {
            "sites": len(tr),
            "seconds": round(dt, 2),
            "sites_per_s_wall": len(tr) / dt if dt else None,
            "tiles_expected": len(tr) * dc.tiles_per_site,
            "fg_candidates_rejected": n_rej,
            "fg_candidates_checked": n_cand,
            "fg_frac_median": float(np.median(fg)) if fg else None,
            "norm_mode": dc.norm_mode,
            "tile_size": dc.tile_size,
        }
        idx = w.close(st)
        st["tiles_written"] = len(idx)
        summary["train"] = st
    if a.only in (None, "dev"):
        dv = m[m.dev].reset_index(drop=True)
        w = ShardWriter(dc.dev_shards_dir, "dev", dc.shard_max_bytes, dc.zstd_level)
        t0 = time.time()
        for r, u8 in run_pass(dv, process_dev, dc.s3_workers):
            meta = {
                **site_meta(r),
                "is_anchor": bool(r.is_anchor),
                "plate_split": r.plate_split,
                "eval_dmso": bool(r.eval_dmso),
            }
            w.write(f"{r.source}__{r.plate}__{r.well}__s{int(r.site)}", u8, meta)
        dt = time.time() - t0
        w.close({"sites": len(dv), "seconds": round(dt, 2), "site_size": dc.dev_site_size})
        summary["dev"] = {"sites": len(dv), "seconds": round(dt, 2)}
    summary["timing_thread_seconds"] = {k: round(v, 2) for k, v in timer.t.items()}
    n_sites = summary.get("train", {}).get("sites", 0) + summary.get("dev", {}).get("sites", 0)
    if timer.t.get("process") and n_sites:
        summary["cpu_process_s_per_site_one_thread"] = timer.t["process"] / n_sites
        summary["read_decode_illum_s_per_site_one_thread"] = timer.t["read"] / n_sites
    summary["wall_seconds"] = round(time.time() - t_all, 2)
    summary["bytes_downloaded"] = g.bytes_downloaded
    summary["illum_headers"] = il.header_info
    out = Path(dc.manifest).with_name("write_shards_summary.json")
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "illum_headers"}, indent=1))


if __name__ == "__main__":
    main()

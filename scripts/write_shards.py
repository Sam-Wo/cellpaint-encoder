"""Build the tile cache (training shards) and the dev cache in one pass (DESIGN.md 1.3).

    python scripts/write_shards.py --config configs/full.yaml [--only norm|train|dev] [--procs 8]

1. per-plate DMSO percentiles from the manifest's ``norm`` sites -> ``plate_norm.json``
2. training tiles: illum-corrected, plate-normalised uint8, 2 x 448^2 per site with the Otsu
   DNA foreground filter -> ``data.shards_dir``
3. dev cache: anchor wells + 4 DMSO wells per plate of the dev sources, centred 896^2 sites
   -> ``data.dev_shards_dir``

Parallelism: ``--procs`` worker processes (decode + illum + normalise + tile are CPU-bound and
hold the GIL, so threads alone plateau at ~2x one core), each with ``data.s3_workers / procs``
fetch threads. Work is cut into plate-contiguous chunks and written in chunk order, so the output
is deterministic given the manifest, the code and ``data.manifest_seed``. (Shards are not
plate-mixed; the sampler reads by index, so shard order does not affect training batches.)
``data.byte_cap`` is enforced per worker process. Timing, the zstd ratio and the foreground drop
rate go into ``stats.json`` and ``write_shards_summary.json``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.gallery import Gallery
from cellpaint_encoder.data.manifest import _rng, check_train_index
from cellpaint_encoder.data.normalize import normalise_site
from cellpaint_encoder.data.plate_norm import compute_plate_norms
from cellpaint_encoder.data.shards import ShardWriter
from cellpaint_encoder.data.sites import IllumCache, read_site, read_site_uint8
from cellpaint_encoder.data.tiles import centre_crop, training_tiles

META_COLS = ["source", "batch", "plate", "well", "site", "row", "col", "compound_id", "is_dmso"]
_W: dict = {}  # per-process worker state


def site_meta(r: dict) -> dict:
    d = {c: r[c] for c in META_COLS}
    d["site"], d["row"], d["col"] = int(d["site"]), int(d["row"]), int(d["col"])
    d["is_dmso"] = bool(d["is_dmso"])
    return d


def _init(cfg_path, overrides, norms, threads):
    cfg = load_config(cfg_path, overrides)
    g = Gallery(mirror=cfg.data.raw_mirror, byte_cap=cfg.data.byte_cap, ledger=False)
    _W.update(cfg=cfg, g=g, il=IllumCache(g, max_plates=2), norms=norms, threads=threads)


def _process_site(kind: str, r: dict):
    dc, g, il, norms = _W["cfg"].data, _W["g"], _W["il"], _W["norms"]
    row = pd.Series(r)
    t0 = time.time()
    if dc.norm_mode == "per_plate_dmso" or kind == "dev":
        nm = norms[r["plate"]]
        u8 = read_site_uint8(g, il, row, nm["lo"], nm["hi"])  # fused, cache-friendly path
    else:
        u8 = normalise_site(read_site(g, il, row), dc.norm_mode, None, dc.pct_lo, dc.pct_hi)
    t1 = time.time()
    if kind == "train":
        rng = _rng(dc.manifest_seed, "tiles", r["plate"], r["well"], str(r["site"]))
        out, info = training_tiles(u8, dc.tile_size, dc.tiles_per_site, dc.fg_min, rng)
    else:
        out, info = [np.ascontiguousarray(centre_crop(u8, dc.dev_site_size))], {}
    return r, out, info, t1 - t0, time.time() - t1


def _process_chunk(args):
    kind, rows = args
    before = _W["g"].bytes_downloaded
    with ThreadPoolExecutor(_W["threads"]) as ex:
        res = list(ex.map(lambda r: _process_site(kind, r), rows))
    return res, _W["g"].bytes_downloaded - before


def chunks(df: pd.DataFrame, size: int):
    """Plate-contiguous chunks (a worker loads each plate's illumination once)."""
    for _, g in df.groupby("plate", sort=True):
        recs = g.to_dict("records")
        for i in range(0, len(recs), size):
            yield recs[i : i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", choices=["norm", "train", "dev"], default=None)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dc = cfg.data
    if dc.resample_um_per_px is not None:
        raise NotImplementedError("data.resample_um_per_px (R7) is not implemented yet; see RUNBOOK 4")
    m = pd.read_parquet(dc.manifest)
    g = Gallery(mirror=dc.raw_mirror, byte_cap=dc.byte_cap)
    il = IllumCache(g, max_plates=4)
    norm_path = Path(dc.manifest).with_name("plate_norm.json")
    t_all = time.time()
    norms = compute_plate_norms(m, g, il, norm_path, dc.pct_lo, dc.pct_hi, dc.manifest_seed, dc.s3_workers)
    summary: dict = {
        "plate_norm_plates": len(norms),
        "norm_seconds": round(time.time() - t_all, 2),
        "procs": a.procs,
    }
    need = set(m[m.train | m.dev].plate) - set(norms)
    if need and dc.norm_mode == "per_plate_dmso":
        raise RuntimeError(f"plates without normalisation sites: {sorted(need)[:5]}")
    if a.only == "norm":
        print(json.dumps(summary, indent=1))
        return
    threads = max(1, dc.s3_workers // max(a.procs, 1))
    t_read = t_proc = 0.0
    net_bytes = g.bytes_downloaded
    with ProcessPoolExecutor(
        a.procs, initializer=_init, initargs=(a.config, a.overrides, norms, threads)
    ) as ex:
        for kind, sel, out_dir in (("train", m.train, dc.shards_dir), ("dev", m.dev, dc.dev_shards_dir)):
            if a.only not in (None, kind):
                continue
            rows = m[sel].reset_index(drop=True)
            w = ShardWriter(out_dir, kind, dc.shard_max_bytes, dc.zstd_level)
            t0, n_rej, n_cand, fg = time.time(), 0, 0, []
            for res, nb in ex.map(_process_chunk, ((kind, c) for c in chunks(rows, a.chunk))):
                net_bytes += nb
                for r, out, info, tr_, tp_ in res:
                    t_read += tr_
                    t_proc += tp_
                    stem = f"{r['source']}__{r['plate']}__{r['well']}__s{int(r['site'])}"
                    if kind == "train":
                        n_rej += info["rejected"]
                        n_cand += len(info["fg_fracs"])
                        fg += info["fg_fracs"]
                        for k, t in enumerate(out):
                            w.write(f"{stem}__t{k}", t, {**site_meta(r), "tile": k})
                    else:
                        meta = {
                            **site_meta(r),
                            "is_anchor": bool(r["is_anchor"]),
                            "plate_split": r["plate_split"],
                            "eval_dmso": bool(r["eval_dmso"]),
                        }
                        w.write(stem, out[0], meta)
            dt = time.time() - t0
            st = {
                "sites": len(rows),
                "seconds": round(dt, 2),
                "sites_per_s_wall": len(rows) / dt if dt else None,
            }
            if kind == "train":
                st.update(
                    {
                        "tiles_expected": len(rows) * dc.tiles_per_site,
                        "fg_candidates_rejected": n_rej,
                        "fg_candidates_checked": n_cand,
                        "fg_frac_median": float(np.median(fg)) if fg else None,
                        "norm_mode": dc.norm_mode,
                        "tile_size": dc.tile_size,
                    }
                )
            else:
                st["site_size"] = dc.dev_site_size
            idx = w.close(st)
            if kind == "train":
                check_train_index(idx, m, dc)  # leakage check on what was written (R8)
            st["samples_written"] = len(idx)
            summary[kind] = st
    n_sites = sum(summary.get(k, {}).get("sites", 0) for k in ("train", "dev"))
    if n_sites:
        summary["read_decode_illum_normalise_s_per_site_one_core"] = t_read / n_sites
        summary["tile_filter_s_per_site_one_core"] = t_proc / n_sites
    summary["wall_seconds"] = round(time.time() - t_all, 2)
    summary["bytes_downloaded"] = net_bytes
    summary["illum_headers_sample"] = dict(list(il.header_info.items())[:10])
    out = Path(dc.manifest).with_name("write_shards_summary.json")
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "illum_headers_sample"}, indent=1))


if __name__ == "__main__":
    main()

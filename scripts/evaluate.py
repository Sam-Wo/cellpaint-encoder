"""Evaluate representations on identical wells with identical post-processing (DESIGN.md 4).

    py -3.13 -I -X utf8 scripts/evaluate.py --config configs/full.yaml \
        --rep ours=runs/embeddings/final/ours/sites.parquet \
        --rep imagenet_dino=runs/embeddings/final/imagenet_dino/sites.parquet \
        --cellprofiler-gallery --out results/dev

CellProfiler: ``--cellprofiler-gallery`` fetches the per-plate profiles of the evaluation wells
from the gallery (the baseline proper); ``--cellprofiler-wells <parquet>`` takes a precomputed
well table instead (e.g. ShapeShift's wells, for continuity checks / smoke tests).

source_6 stays locked unless ``eval.unlock_heldout=true`` (final scoring, once, all methods).
Comparisons: ours vs CellProfiler (P1) and vs ImageNet-frozen (P2) on per-compound AP, with the
compound bootstrap and one-sided Wilcoxon; M1 non-inferiority (P3); Holm over the secondaries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json

import numpy as np
import pandas as pd

from cellpaint_encoder.config import load_config, to_dict
from cellpaint_encoder.eval import stats
from cellpaint_encoder.eval.aggregate import KEYS, sites_to_wells
from cellpaint_encoder.eval.pipeline import evaluate_wells, public
from cellpaint_encoder.eval.report import write_report


def eval_well_meta(m: pd.DataFrame, sets) -> pd.DataFrame:
    w = m[m.eval_set.isin(sets)]
    cols = [
        c
        for c in [
            "source",
            "batch",
            "plate",
            "well",
            "row",
            "col",
            "compound_id",
            "is_dmso",
            "is_anchor",
            "eval_dmso",
            "plate_split",
            "eval_set",
        ]
        if c in w
    ]
    return w[cols].drop_duplicates(KEYS[:1] + KEYS[2:]).reset_index(drop=True)


def cellprofiler_rep(a, cfg, m, sets):
    from cellpaint_encoder.eval.cellprofiler import from_profiles, prepare_cellprofiler

    wm = eval_well_meta(m, sets)
    if a.cellprofiler_wells:
        w = pd.read_parquet(a.cellprofiler_wells)
        feats = [c for c in w.columns if c.startswith("f") and c[1:].isdigit()]
        j = wm.merge(w[["source", "plate", "well", *feats]], on=["source", "plate", "well"], how="inner")
        meta, X = j.drop(columns=feats), j[feats].to_numpy(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        return (
            meta,
            X,
            {
                "source": "precomputed wells (already selected/sphered): continuity/smoke only",
                "n_features": len(feats),
            },
        )
    from cellpaint_encoder.data.gallery import Gallery
    from cellpaint_encoder.eval.cellprofiler import fetch_profiles

    g = Gallery(mirror=cfg.data.raw_mirror, byte_cap=cfg.data.byte_cap)
    prof = fetch_profiles(g, wm)
    meta, X, feats = from_profiles(prof, wm)
    dev = meta.source.isin(list(cfg.eval.dev_query_sources)).to_numpy() & meta.eval_dmso.to_numpy().astype(
        bool
    )
    X, sel, info = prepare_cellprofiler(
        meta, X, feats, meta.eval_dmso.to_numpy().astype(bool), dev, cfg.eval.clip_z
    )
    info["bytes_downloaded"] = g.bytes_downloaded
    return meta, X, info


def per_compound_primary(res: dict) -> pd.Series:
    if "primary" in res["_tables"]:
        t = res["_tables"]["primary"]
        return t.set_index("compound_id").mAP
    t = res["_tables"].get("dev_cross_source", pd.DataFrame())
    if t.empty:
        return pd.Series(dtype=float)
    return t.set_index(t.query_source + "|" + t.compound_id).mAP


def per_unit(res: dict, name: str) -> pd.Series:
    t = res["_tables"].get(name, pd.DataFrame())
    if t.empty:
        return pd.Series(dtype=float)
    return t.set_index(t.source + "|" + t.compound_id).mAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rep", action="append", default=[], help="name=path/to/sites.parquet")
    ap.add_argument("--cellprofiler-gallery", action="store_true")
    ap.add_argument("--cellprofiler-wells", default=None)
    ap.add_argument("--sets", default="E1")
    ap.add_argument("--ours", default="ours", help="name of our representation in --rep")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    sets = a.sets.split(",")
    m = pd.read_parquet(cfg.data.manifest)
    moa = (
        pd.read_csv(cfg.eval.moa_table) if cfg.eval.moa_table and Path(cfg.eval.moa_table).exists() else None
    )
    reps: dict[str, tuple] = {}
    for spec in a.rep:
        name, _, path = spec.partition("=")
        sites = pd.read_parquet(path)
        sites = sites[sites.eval_set.isin(sets)] if "eval_set" in sites else sites
        meta, X = sites_to_wells(sites)
        reps[name] = (meta, X, {"sites_file": path})
    if a.cellprofiler_gallery or a.cellprofiler_wells:
        reps["cellprofiler"] = cellprofiler_rep(a, cfg, m, sets)
    # identical wells: intersect the well sets of all representations
    key = lambda d: set(d.source + "|" + d.plate + "|" + d.well)  # noqa: E731
    common = set.intersection(*(key(r[0]) for r in reps.values())) if reps else set()
    results, full = {}, {}
    for name, (meta, X, info) in reps.items():
        k = (meta.source + "|" + meta.plate + "|" + meta.well).isin(common).to_numpy()
        res = evaluate_wells(meta[k].reset_index(drop=True), X[k], cfg, moa=moa)
        res["input"] = info
        full[name] = res
        results[name] = public(res)
        print(
            f"{name}: wells={res['n_wells']} eps={res['eps']} dev_xsrc_mAP={res['dev_cross_source']['mAP']:.4f} "
            f"M1={res['M1']['mAP']:.4f} M2={res['M2']['mAP']:.4f}"
        )
    comps = {}
    if a.ours in full:
        o = full[a.ours]
        for other, label in (("cellprofiler", "P1_vs_cellprofiler"), ("imagenet_dino", "P2_vs_imagenet")):
            if other in full:
                comps[label] = stats.paired_compound_test(
                    per_compound_primary(o),
                    per_compound_primary(full[other]),
                    n_boot=cfg.eval.n_boot,
                    seed=cfg.eval.seed,
                )
        if "cellprofiler" in full:
            comps["P3_M1_noninferiority_vs_cellprofiler"] = stats.noninferiority(
                per_unit(o, "M1"),
                per_unit(full["cellprofiler"], "M1"),
                n_boot=cfg.eval.n_boot,
                seed=cfg.eval.seed,
            )
            comps["P4_position_inflation"] = {
                "ours": o["position_inflation_M2_minus_M1"],
                "cellprofiler": full["cellprofiler"]["position_inflation_M2_minus_M1"],
            }
        fam = {k: v.get("p_wilcoxon", np.nan) for k, v in comps.items() if k.startswith(("P2", "P3"))}
        comps["holm_secondaries"] = stats.holm(fam) if fam else {}
    out = write_report(
        results, a.out, {"comparisons": comps, "config": to_dict(cfg), "common_wells": len(common)}
    )
    for name, res in full.items():
        for t, df in res["_tables"].items():
            if isinstance(df, pd.DataFrame) and len(df):
                (out / "tables").mkdir(exist_ok=True)
                df.to_csv(out / "tables" / f"{name}__{t}.csv", index=False)
    print(json.dumps(comps, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

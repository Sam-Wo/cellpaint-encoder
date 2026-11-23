"""Evaluation pipeline: identical wells, aggregation and post-processing for every representation.

``source_6`` is **locked** (DESIGN.md 4.1): unless ``eval.unlock_heldout`` is true, every row of
the held-out source is dropped before any statistic is computed, and the primary metric runs in
its dev form (each dev source queried against the other dev sources). The final scoring sets
``eval.unlock_heldout=true`` once, for all methods together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from . import probes, retrieval, shapeshift_hook
from .aggregate import sites_to_wells
from .profiles import postprocess, robust_z
from .retrieval import unit_rows


def lock_heldout(meta: pd.DataFrame, X: np.ndarray, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    if cfg.eval.unlock_heldout:
        return meta, X
    keep = (meta["source"] != cfg.data.heldout_source).to_numpy()
    return meta[keep].reset_index(drop=True), X[keep]


def diffpos_auc(meta: pd.DataFrame, X: np.ndarray, sources) -> float:
    """Replicate-pair vs other-compound-pair cosine AUC over pairs at different (source, well)
    positions, anchor wells of ``sources`` (the dev rule that picks eps; ShapeShift's ``diff_pos``)."""
    m = meta["is_anchor"].to_numpy().astype(bool) & meta["source"].isin(list(sources)).to_numpy()
    if m.sum() < 3:
        return float("nan")
    d = meta[m]
    U = unit_rows(X[m].astype(np.float64))
    a, b = np.triu_indices(len(U), 1)
    s = np.einsum("ij,ij->i", U[a], U[b])
    cpd, src, well = (d[c].astype(str).to_numpy() for c in ("compound_id", "source", "well"))
    diff = ~((src[a] == src[b]) & (well[a] == well[b]))
    same = cpd[a] == cpd[b]
    pos, neg = s[diff & same], s[diff & ~same]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ref_mask(meta: pd.DataFrame) -> np.ndarray:
    """Per-plate z-score / whitening reference: the plate's evaluation DMSO wells (16 per plate)."""
    col = "eval_dmso" if "eval_dmso" in meta else "is_dmso"
    return meta[col].to_numpy().astype(bool)


def choose_eps(meta: pd.DataFrame, X: np.ndarray, cfg: Config) -> tuple[float, dict]:
    """eps per representation from the grid, by dev different-position AUC (dev sources only);
    ties go to the larger eps (more regularised)."""
    e = cfg.eval
    dev = meta["source"].isin(list(e.dev_query_sources) + list(e.dev_reference_sources)).to_numpy()
    md, Xd = meta[dev].reset_index(drop=True), X[dev]
    table = {}
    for eps in e.eps_grid:
        V, _ = postprocess(md, Xd, ref_mask(md), eps, e.var_keep, e.clip_z, e.mad_floor_rel, e.mad_floor_abs)
        table[eps] = diffpos_auc(md, V, set(e.dev_query_sources) | set(e.dev_reference_sources))
    finite = {k: v for k, v in table.items() if np.isfinite(v)}
    if not finite:
        return float(e.eps_grid[len(e.eps_grid) // 2]), {
            "auc_by_eps": table,
            "note": "AUC undefined at this size; middle of grid",
        }
    best = max(finite.values())
    eps = max(k for k, v in finite.items() if v >= best - 1e-12)
    return float(eps), {"auc_by_eps": table}


def evaluate_wells(
    meta: pd.DataFrame,
    X: np.ndarray,
    cfg: Config,
    eps: float | None = None,
    moa: pd.DataFrame | None = None,
    which: tuple[str, ...] = ("all",),
) -> dict:
    e = cfg.eval
    meta, X = lock_heldout(meta.reset_index(drop=True), X, cfg)
    res: dict = {"n_wells": len(meta), "n_plates": int(meta.plate.nunique()), "locked": not e.unlock_heldout}
    if eps is None:
        eps, info = choose_eps(meta, X, cfg)
        res["eps_selection"] = info
    res["eps"] = eps
    V, pinfo = postprocess(
        meta, X, ref_mask(meta), eps, e.var_keep, e.clip_z, e.mad_floor_rel, e.mad_floor_abs
    )
    res["postprocess"] = pinfo
    tables: dict[str, pd.DataFrame] = {}
    want = set(which)
    every = "all" in want

    # ---- primary (4.2): held-out form only when unlocked, dev form otherwise
    if every or "primary" in want:
        if e.unlock_heldout:
            per, s, _ = retrieval.cross_source(
                meta, V, [e.query_source], e.reference_sources, n_null=e.n_null, seed=e.seed
            )
            per_s, s_s, _ = retrieval.cross_source(
                meta, V, [e.query_source], e.reference_sources, strict=True, n_null=e.n_null, seed=e.seed
            )
            res["primary"], res["primary_strict"] = s, s_s
            tables["primary"], tables["primary_strict"] = per, per_s
        frames, sums = [], []
        for qs in e.dev_query_sources:
            refs = [r for r in e.dev_reference_sources if r != qs]
            if not refs:
                continue
            per, s, _ = retrieval.cross_source(meta, V, [qs], refs, n_null=e.n_null, seed=e.seed)
            frames.append(per.assign(query_source=qs))
            sums.append(s)
        dev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        tables["dev_cross_source"] = dev
        res["dev_cross_source"] = {
            "mAP": float(dev.mAP.mean()) if len(dev) else float("nan"),
            "n_groups": len(dev),
            "by_query_source": {qs: s["mAP"] for qs, s in zip(e.dev_query_sources, sums)},
        }

    # ---- M1 / M2 (+ DMSO-negative variant)
    if every or "m1" in want:
        src_m = sorted(
            set(meta.source) & (set(e.reference_sources) | set(e.dev_query_sources) | {e.query_source})
        )
        for name, kw in (
            ("M1", dict(position_restricted=True)),
            ("M2", dict(position_restricted=False)),
            ("M1_dmso_neg", dict(position_restricted=True, dmso_negatives=True)),
        ):
            per, s, _ = retrieval.within_source(meta, V, src_m, n_null=e.n_null, seed=e.seed, **kw)
            res[name], tables[name] = s, per
        res["position_inflation_M2_minus_M1"] = res["M2"]["mAP"] - res["M1"]["mAP"]

    # ---- M3 MoA
    if (every or "m3" in want) and moa is not None:
        cons, C = retrieval.consensus(meta, V)
        lab = cons.compound_id.map(dict(zip(moa.compound_id, moa.moa))).to_numpy()
        per, s, _ = retrieval.label_retrieval(lab, C, n_null=e.n_null, seed=e.seed)
        res["M3_moa"], tables["M3_moa"] = s, per

    # ---- M6: training-source queries on held-out plates vs seen plates
    if (every or "m6" in want) and "plate_split" in meta:
        for split in ("heldout", "train"):
            qm = (meta["plate_split"] == split).to_numpy()
            if not qm.any():
                res[f"M6_{split}_plates"] = {"mAP": float("nan"), "note": "no such plates"}
                continue
            frames = []
            for qs in e.dev_query_sources:
                refs = [r for r in e.reference_sources if r != qs]
                per, _, _ = retrieval.cross_source(
                    meta, V, [qs], refs, n_null=e.n_null, seed=e.seed, query_mask=qm
                )
                frames.append(per)
            per = pd.concat(frames)
            res[f"M6_{split}_plates"] = {
                "mAP": float(per.mAP.mean()) if len(per) else float("nan"),
                "n_groups": len(per),
            }

    # ---- diagnostics (4.4): before and after post-processing
    if every or "probes" in want:
        dm = meta["is_dmso"].to_numpy().astype(bool)
        Zraw, _ = robust_z(meta, X, ref_mask(meta), e.clip_z, e.mad_floor_rel, e.mad_floor_abs)
        res["position_probe"] = probes.position_probe(meta, V, dm, seed=e.seed)
        res["same_well_gap"] = probes.same_well_gap(meta, V, dm)
        res["source_probe_raw"] = probes.source_probe(meta, X, dm, seed=e.seed)
        res["source_probe_z"] = probes.source_probe(meta, Zraw, dm, seed=e.seed)
        res["source_probe_post"] = probes.source_probe(meta, V, dm, seed=e.seed)

    if every or "shapeshift" in want:
        _, s = shapeshift_hook.strict_replicate_reproducibility(
            meta, V, n_null=min(e.n_null, 2000), seed=e.seed
        )
        res["shapeshift_strict"] = s
    res["_tables"] = tables
    res["_V"] = V
    res["_meta"] = meta
    return res


def evaluate_sites(sites: pd.DataFrame, cfg: Config, **kw) -> dict:
    meta, X = sites_to_wells(sites)
    return evaluate_wells(meta, X, cfg, **kw)


def public(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def dev_eval_hook(trainer) -> dict:
    """Called by the trainer every ``eval.dev_every_epochs``: embed the dev cache with the EMA
    teacher backbone and report dev cross-source mAP, M1 and the position probe."""
    import copy

    from ..embed.encoders import ViTEncoder
    from ..embed.run import embed_dev_shards

    cfg = trainer.cfg
    bb = copy.deepcopy(trainer.teacher.backbone).eval()
    enc = ViTEncoder(
        "ours_dev",
        bb,
        trainer.channel_mean,
        trainer.channel_std,
        trainer.device,
        crop=cfg.eval.crop,
        grid=cfg.eval.grid,
        amp=trainer.use_amp and trainer.device.type == "cuda",
    )
    sites = embed_dev_shards(enc, cfg.data.dev_shards_dir, cfg.eval.batch_size)
    r = evaluate_sites(sites, cfg, which=("primary", "m1", "probes"))
    return {
        "cross_source_mAP": r["dev_cross_source"]["mAP"],
        "M1_mAP": r["M1"]["mAP"],
        "M2_mAP": r["M2"]["mAP"],
        "eps": r["eps"],
        "pos_probe_row": r["position_probe"]["bal_acc_row"],
        "pos_probe_col": r["position_probe"]["bal_acc_col"],
    }

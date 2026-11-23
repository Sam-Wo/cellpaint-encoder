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
    """All metrics for one representation. Post-processing (robust z, whitening) is fitted once on
    every evaluation plate present; the E1 metrics use E1 wells only, M5 adds E2 (low dose) queries,
    M4 uses E3 (Target-2) wells, and the E4 (all-DMSO plates) position probe uses E4 wells."""
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
    sets = meta["eval_set"].astype(str).to_numpy() if "eval_set" in meta else np.full(len(meta), "E1")
    e1 = sets == "E1"
    m1, V1, X1 = meta[e1].reset_index(drop=True), V[e1], X[e1]
    q_kw = dict(n_null=e.n_null, seed=e.seed)

    # ---- primary (4.2): held-out form only when unlocked, dev form otherwise
    if every or "primary" in want:
        if e.unlock_heldout:
            per, s, _ = retrieval.cross_source(m1, V1, [e.query_source], e.reference_sources, **q_kw)
            per_s, s_s, _ = retrieval.cross_source(
                m1, V1, [e.query_source], e.reference_sources, strict=True, **q_kw
            )
            res["primary"], res["primary_strict"] = s, s_s
            tables["primary"], tables["primary_strict"] = per, per_s
        frames, sums = [], {}
        for qs in e.dev_query_sources:
            refs = [r for r in e.dev_reference_sources if r != qs]
            if not refs:
                continue
            per, s, _ = retrieval.cross_source(m1, V1, [qs], refs, **q_kw)
            frames.append(per.assign(query_source=qs))
            sums[qs] = s["mAP"]
        dev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        tables["dev_cross_source"] = dev
        res["dev_cross_source"] = {
            "mAP": float(dev.mAP.mean()) if len(dev) else float("nan"),
            "n_groups": len(dev),
            "by_query_source": sums,
        }

    # ---- M1 / M2 (+ DMSO-negative variant) on sources 2, 3, 6, 8 (4.3)
    src_m = sorted(set(m1.source) & set(e.m1_sources))
    if every or "m1" in want:
        for name, kw in (
            ("M1", dict(position_restricted=True)),
            ("M2", dict(position_restricted=False)),
            ("M1_dmso_neg", dict(position_restricted=True, dmso_negatives=True)),
        ):
            per, s, _ = retrieval.within_source(m1, V1, src_m, **q_kw, **kw)
            res[name], tables[name] = s, per
        res["position_inflation_M2_minus_M1"] = res["M2"]["mAP"] - res["M1"]["mAP"]

    # ---- M3 MoA on consensus profiles (median within source, mean across sources 2, 3, 6, 8)
    if (every or "m3" in want) and moa is not None:
        cons, C = retrieval.consensus(m1, V1, sources=src_m)
        lab = cons.compound_id.map(dict(zip(moa.compound_id, moa.moa))).to_numpy()
        per, s, q = retrieval.label_retrieval(lab, C, **q_kw)
        q["compound_id"] = cons.compound_id.to_numpy()[q["query"].to_numpy()] if len(q) else []
        res["M3_moa"], tables["M3_moa"], tables["M3_moa_queries"] = s, per, q

    # ---- M4: Target-2 cross-source compound retrieval (E3), each source against the others
    if (every or "m4" in want) and (sets == "E3").any():
        m4 = meta[sets == "E3"].copy()
        m4["is_anchor"] = ~m4.is_dmso.astype(bool) & (m4.compound_id.astype(str) != "")
        V4 = V[sets == "E3"]
        frames = []
        for qs in sorted(set(m4.source)):
            refs = [r for r in set(m4.source) if r != qs]
            if refs:
                per, _, _ = retrieval.cross_source(m4.reset_index(drop=True), V4, [qs], refs, **q_kw)
                frames.append(per.assign(query_source=qs))
        per = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        tables["M4_target2"] = per
        res["M4_target2"] = {"mAP": float(per.mAP.mean()) if len(per) else float("nan"), "n_groups": len(per)}

    # ---- M5: cross-dose (exploratory): low-dose (E2) queries against 10 uM references (E1)
    if (every or "m5" in want) and (sets == "E2").any():
        m5 = (sets == "E1") | (sets == "E2")
        per, s, _ = retrieval.cross_source(
            meta[m5].reset_index(drop=True), V[m5], [e.low_dose_source], e.reference_sources, **q_kw
        )
        res["M5_cross_dose"], tables["M5_cross_dose"] = s, per

    # ---- M6: training-source queries on held-out plates vs seen plates
    if (every or "m6" in want) and "plate_split" in m1:
        for split in ("heldout", "train"):
            qm = (m1["plate_split"] == split).to_numpy()
            if not qm.any():
                res[f"M6_{split}_plates"] = {"mAP": float("nan"), "note": "no such plates"}
                continue
            frames = []
            for qs in e.dev_query_sources:
                refs = [r for r in e.reference_sources if r != qs]
                per, _, _ = retrieval.cross_source(m1, V1, [qs], refs, query_mask=qm, **q_kw)
                frames.append(per)
            per = pd.concat(frames)
            res[f"M6_{split}_plates"] = {
                "mAP": float(per.mAP.mean()) if len(per) else float("nan"),
                "n_groups": len(per),
            }

    # ---- diagnostics (4.4): E1 DMSO wells, before and after post-processing; E4 position probe
    if every or "probes" in want:
        dm = m1["is_dmso"].to_numpy().astype(bool)
        Zraw, _ = robust_z(m1, X1, ref_mask(m1), e.clip_z, e.mad_floor_rel, e.mad_floor_abs)
        res["position_probe"] = probes.position_probe(m1, V1, dm, seed=e.seed)
        res["same_well_gap"] = probes.same_well_gap(m1, V1, dm)
        res["source_probe_raw"] = probes.source_probe(m1, X1, dm, seed=e.seed)
        res["source_probe_z"] = probes.source_probe(m1, Zraw, dm, seed=e.seed)
        res["source_probe_post"] = probes.source_probe(m1, V1, dm, seed=e.seed)
        if (sets == "E4").any():
            m4p = meta[sets == "E4"].reset_index(drop=True)
            res["position_probe_E4"] = probes.position_probe(m4p, V[sets == "E4"], seed=e.seed)

    if every or "shapeshift" in want:
        sel = m1.source.isin(src_m).to_numpy()
        _, s = shapeshift_hook.strict_replicate_reproducibility(
            m1[sel], V1[sel], n_null=min(e.n_null, 2000), seed=e.seed
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

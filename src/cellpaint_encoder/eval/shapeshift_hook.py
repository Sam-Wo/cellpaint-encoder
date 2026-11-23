"""ShapeShift hook (DESIGN.md 4.7) on the 175-compound anchor set.

(i)  strict replicate reproducibility: mean cosine over a compound's **different-position**
     replicate pairs (not same source + well) against an other-compound null (each replicate well
     replaced by a well of a different anchor compound from the same source); ``active_strict`` if
     p < 0.05. ShapeShift's CellProfiler value: 73.7 % (74 %) at 10 uM.
(ii) M3 MoA mAP: :func:`retrieval.label_retrieval` on consensus profiles.
(iii) exploratory Mantel vs Tahoe: :func:`export_for_compare_tiers` writes the embedding in the
     layout ShapeShift's ``step2/compare_tiers.py`` reads (``morph_profiles_high.parquet``,
     ``step2/morph_wells_high.parquet``); rerun that script on a scratch copy of the ShapeShift tree
     with these files swapped in. No pre-registered claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval import consensus, unit_rows


def strict_replicate_reproducibility(
    meta: pd.DataFrame, X: np.ndarray, n_null: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    a = meta["is_anchor"].to_numpy().astype(bool)
    d = meta[a].reset_index(drop=True)
    U = unit_rows(X[a].astype(np.float64))
    S = U @ U.T
    src, well, cpd = (d[c].astype(str).to_numpy() for c in ("source", "well", "compound_id"))
    rows = []
    for c in np.unique(cpd):
        idx = np.flatnonzero(cpd == c)
        if len(idx) < 2:
            continue
        ii, jj = np.triu_indices(len(idx), 1)
        qi, qj = idx[ii], idx[jj]
        diffpos = ~((src[qi] == src[qj]) & (well[qi] == well[qj]))
        basis = "diffpos" if diffpos.any() else "allpairs"
        qi, qj = (qi[diffpos], qj[diffpos]) if diffpos.any() else (qi, qj)
        obs = float(S[qi, qj].mean())
        # null: replace each replicate well by a random other-compound well of the same source
        pools = {s: np.flatnonzero((src == s) & (cpd != c)) for s in np.unique(src[idx])}
        if any(len(p) == 0 for p in pools.values()):
            rows.append({"compound_id": c, "obs": obs, "p": np.nan, "basis": basis, "n_pairs": len(qi)})
            continue
        sub = {k: v for k, v in zip(idx, range(len(idx)))}
        draws = np.stack([rng.choice(pools[src[k]], n_null) for k in idx], axis=1)  # [n_null, n_wells]
        ni, nj = draws[:, [sub[k] for k in qi]], draws[:, [sub[k] for k in qj]]
        null = S[ni, nj].mean(1)
        p = (1 + np.sum(null >= obs)) / (1 + n_null)
        rows.append({"compound_id": c, "obs": obs, "p": p, "basis": basis, "n_pairs": len(qi)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"frac_active_strict": float("nan"), "n_compounds": 0}
    df["active_strict"] = df.p < alpha
    dp = df[df.basis == "diffpos"]
    return df, {
        "frac_active_strict": float(df.active_strict.mean()),
        "n_compounds": len(df),
        "frac_active_strict_diffpos_only": float(dp.active_strict.mean()) if len(dp) else float("nan"),
        "n_diffpos": len(dp),
        "shapeshift_cellprofiler_ref": 0.737,
    }


def export_for_compare_tiers(
    meta: pd.DataFrame, X: np.ndarray, out_dir: str | Path, dose: str = "high"
) -> dict:
    """Write ShapeShift-format files: consensus per compound (``compound_id, f0..fN``) and wells
    (``compound_id, jcp, source, batch, plate, well, is_dmso, f0..fN``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cons, C = consensus(meta, X)
    fcols = [f"f{i}" for i in range(X.shape[1])]
    prof = pd.concat([cons[["compound_id"]], pd.DataFrame(C.astype(np.float32), columns=fcols)], axis=1)
    prof.to_parquet(out / f"morph_profiles_{dose}.parquet", index=False)
    w = meta[["compound_id", "source", "batch", "plate", "well", "is_dmso"]].copy()
    w["jcp"] = np.where(w.is_dmso, "JCP2022_033924", w.compound_id)
    w = pd.concat([w.reset_index(drop=True), pd.DataFrame(X.astype(np.float32), columns=fcols)], axis=1)
    w.to_parquet(out / f"morph_wells_{dose}.parquet", index=False)
    return {
        "profiles": str(out / f"morph_profiles_{dose}.parquet"),
        "wells": str(out / f"morph_wells_{dose}.parquet"),
        "n_compounds": len(prof),
    }

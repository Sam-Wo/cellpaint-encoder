"""Diagnostics (DESIGN.md 4.4): well-position probe, same-well similarity gap, source/batch probe.

All are comparisons across methods, not pass/fail. Every function returns ``NaN`` plus a
``note`` when there is too little data (e.g. fewer than 2 plates for GroupKFold).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .retrieval import unit_rows

C_GRID = (0.01, 0.1, 1.0, 10.0)


def _fit_predict(X, y, groups, n_splits=5, inner_splits=3, proba=False, seed=0):
    """Outer GroupKFold predictions; C chosen by inner GroupKFold balanced accuracy."""
    ug = np.unique(groups)
    if len(ug) < 2:
        return None, "fewer than 2 groups (plates)"
    if len(np.unique(y)) < 2:
        return None, "a single class"
    outer = GroupKFold(n_splits=min(n_splits, len(ug)))
    pred = np.empty(len(y), dtype=float if proba else object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        for tr, te in outer.split(X, y, groups):
            best_c, best = C_GRID[1], -1.0
            ug_tr = np.unique(groups[tr])
            if len(ug_tr) >= 2 and len(C_GRID) > 1:
                inner = GroupKFold(n_splits=min(inner_splits, len(ug_tr)))
                for c in C_GRID:
                    sc = []
                    for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                        if len(np.unique(y[tr][itr])) < 2:
                            continue
                        m = _model(c, seed).fit(X[tr][itr], y[tr][itr])
                        sc.append(balanced_accuracy_score(y[tr][ite], m.predict(X[tr][ite])))
                    if sc and np.mean(sc) > best:
                        best, best_c = float(np.mean(sc)), c
            if len(np.unique(y[tr])) < 2:
                pred[te] = y[tr][0] if not proba else 0.5
                continue
            m = _model(best_c, seed).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
    return pred, ""


def _model(c, seed):
    from sklearn.pipeline import make_pipeline

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))


def position_probe(
    meta: pd.DataFrame,
    X: np.ndarray,
    rows: np.ndarray | None = None,
    n_rows: int = 16,
    n_cols: int = 24,
    seed: int = 0,
) -> dict:
    """Predict row (16 classes) and column (24) with L2 logistic regression, GroupKFold by plate.
    Balanced accuracy vs chance; edge-vs-interior AUC."""
    m = np.ones(len(meta), bool) if rows is None else rows
    d, Xm = meta[m], X[m]
    groups = d["plate"].astype(str).to_numpy()
    out = {
        "n_wells": int(m.sum()),
        "n_plates": int(len(np.unique(groups))),
        "chance_row": 1 / n_rows,
        "chance_col": 1 / n_cols,
    }
    for name, y in (("row", d["row"].to_numpy()), ("col", d["col"].to_numpy())):
        pred, note = _fit_predict(Xm, y, groups, seed=seed)
        out[f"bal_acc_{name}"] = (
            float(balanced_accuracy_score(y, pred.astype(y.dtype))) if pred is not None else float("nan")
        )
        if note:
            out[f"note_{name}"] = note
    edge = (
        ((d["row"] == 0) | (d["row"] == n_rows - 1) | (d["col"] == 0) | (d["col"] == n_cols - 1))
        .to_numpy()
        .astype(int)
    )
    pred, note = _fit_predict(Xm, edge, groups, proba=True, seed=seed)
    out["edge_auc"] = (
        float(roc_auc_score(edge, pred)) if pred is not None and len(np.unique(edge)) == 2 else float("nan")
    )
    if note:
        out["note_edge"] = note
    return out


def same_well_gap(meta: pd.DataFrame, X: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """Mean cosine of DMSO pairs on different plates: same well vs different well, within a source
    and across sources at the same well ID (ShapeShift CP: 0.068 vs 0.002; 0.005 vs -0.001)."""
    m = meta["is_dmso"].to_numpy().astype(bool) if rows is None else rows
    d = meta[m]
    U = unit_rows(X[m].astype(np.float64))
    a, b = np.triu_indices(len(U), 1)
    s = np.einsum("ij,ij->i", U[a], U[b])
    src, plate, well = (d[c].astype(str).to_numpy() for c in ("source", "plate", "well"))
    diffp = plate[a] != plate[b]
    samesrc = src[a] == src[b]
    samew = well[a] == well[b]

    def mean(mask):
        return float(s[mask].mean()) if mask.any() else float("nan")

    return {
        "within_same_well": mean(diffp & samesrc & samew),
        "within_diff_well": mean(diffp & samesrc & ~samew),
        "across_same_well": mean(~samesrc & samew),
        "across_diff_well": mean(~samesrc & ~samew),
        "n_within_same_well_pairs": int((diffp & samesrc & samew).sum()),
        "n_across_same_well_pairs": int((~samesrc & samew).sum()),
    }


def source_probe(meta: pd.DataFrame, X: np.ndarray, rows: np.ndarray | None = None, seed: int = 0) -> dict:
    """Predict source from DMSO wells (GroupKFold by plate) and batch within source; silhouette by source."""
    m = meta["is_dmso"].to_numpy().astype(bool) if rows is None else rows
    d, Xm = meta[m], X[m]
    groups = d["plate"].astype(str).to_numpy()
    y = d["source"].astype(str).to_numpy()
    out = {"n_wells": int(m.sum()), "n_sources": int(len(np.unique(y)))}
    pred, note = _fit_predict(Xm, y, groups, seed=seed)
    out["bal_acc_source"] = (
        float(balanced_accuracy_score(y, pred.astype(str))) if pred is not None else float("nan")
    )
    out["chance_source"] = 1 / max(len(np.unique(y)), 1)
    if note:
        out["note_source"] = note
    accs = []
    for s in np.unique(y):
        ms = y == s
        yb = d["batch"].astype(str).to_numpy()[ms]
        p, _ = _fit_predict(Xm[ms], yb, groups[ms], seed=seed)
        if p is not None:
            accs.append(balanced_accuracy_score(yb, p.astype(str)))
    out["bal_acc_batch_within_source"] = float(np.mean(accs)) if accs else float("nan")
    if len(np.unique(y)) >= 2 and len(y) > len(np.unique(y)):
        out["silhouette_source"] = float(silhouette_score(unit_rows(Xm), y, metric="cosine"))
    else:
        out["silhouette_source"] = float("nan")
    return out

"""Post-processing identical for every representation (DESIGN.md 4.6 steps 5-6), plus the
CellProfiler feature selection ported from ShapeShift ``step2/jump_profiles.py`` (MORPH_SPEC).

5. per plate: robust z against the plate's evaluation DMSO wells,
   ``(x - median) / max(1.4826 * MAD, floor)``, floor = ``mad_floor_rel`` x the median over plates
   of the scale (and at least ``mad_floor_abs``), clipped at +-50;
6. PCA-whitening fitted on DMSO wells of all evaluation plates, keeping 99 % of variance, scaled by
   ``1 / sqrt(eigenvalue + eps)``; ``eps`` chosen per representation on the dev set only.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826


def robust_z(
    meta: pd.DataFrame,
    X: np.ndarray,
    ref_mask: np.ndarray,
    clip: float = 50.0,
    floor_rel: float = 0.01,
    floor_abs: float = 1e-12,
    min_ref: int = 1,
) -> tuple[np.ndarray, dict]:
    """Per-plate robust z of ``X`` against the rows in ``ref_mask`` on the same plate."""
    X = X.astype(np.float64)
    plates = meta["plate"].to_numpy()
    groups = pd.Series(np.arange(len(meta))).groupby(plates).indices
    meds, scales, order = [], [], []
    info = {"plates_low_ref": []}
    for p, idx in groups.items():
        ref = idx[ref_mask[idx]]
        if len(ref) < min_ref:
            raise ValueError(f"plate {p}: {len(ref)} reference (DMSO) wells < {min_ref}")
        if len(ref) < 8:
            info["plates_low_ref"].append(p)
        med = np.median(X[ref], axis=0)
        meds.append(med)
        scales.append(MAD_SCALE * np.median(np.abs(X[ref] - med), axis=0))
        order.append(idx)
    scales = np.array(scales)
    ref_scale = np.median(scales, axis=0)
    floor = np.maximum(floor_rel * ref_scale, floor_abs)
    floored = scales < floor[None]
    info["scale_floored_frac"] = float(floored.mean())
    scales = np.where(floored, floor[None], scales)
    Z = np.empty_like(X)
    for med, sc, idx in zip(meds, scales, order):
        Z[idx] = (X[idx] - med) / sc
    info["maxabs"] = np.abs(Z).max(0)
    np.clip(Z, -clip, clip, out=Z)
    return Z.astype(np.float32), info


def fit_whitening(
    D: np.ndarray, var_keep: float = 0.99, eps: float = 1.0
) -> tuple[np.ndarray, np.ndarray, dict]:
    """PCA whitening fitted on reference rows ``D``. Returns (mu, W, info); apply as ``(X - mu) @ W``."""
    D = D.astype(np.float64)
    mu = D.mean(0)
    if len(D) < 2:
        return (
            mu,
            np.eye(D.shape[1]),
            {"sphere_dim": D.shape[1], "degenerate": "fewer than 2 reference wells"},
        )
    C = np.cov(D - mu, rowvar=False)
    lam, U = np.linalg.eigh(np.atleast_2d(C))
    lam, U = lam[::-1].clip(min=0), U[:, ::-1]
    tot = lam.sum()
    if tot <= 0:
        return mu, np.eye(D.shape[1]), {"sphere_dim": D.shape[1], "degenerate": "zero variance"}
    k = int(np.searchsorted(np.cumsum(lam) / tot, var_keep) + 1)
    k = min(k, len(lam))
    W = U[:, :k] / np.sqrt(lam[:k] + eps)
    info = {"sphere_dim": k, "eig_first_last": [float(lam[0]), float(lam[k - 1])], "n_ref": len(D)}
    if len(D) <= D.shape[1]:
        info["rank_deficient"] = f"{len(D)} reference wells for {D.shape[1]} dims"
    return mu, W, info


def apply_whitening(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return ((X - mu) @ W).astype(np.float32)


def postprocess(
    meta: pd.DataFrame,
    X: np.ndarray,
    ref_mask: np.ndarray,
    eps: float,
    var_keep: float = 0.99,
    clip: float = 50.0,
    floor_rel: float = 0.01,
    floor_abs: float = 1e-12,
    whiten_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    Z, zi = robust_z(meta, X, ref_mask, clip, floor_rel, floor_abs)
    ref = ref_mask if whiten_ref is None else whiten_ref
    mu, W, wi = fit_whitening(Z[ref], var_keep, eps)
    return apply_whitening(Z, mu, W), {"z": {k: v for k, v in zi.items() if k != "maxabs"}, "whiten": wi}


# --------------------------------------------------------------------------- CellProfiler selection
CP_CHANNELS = ["AGP", "DNA", "ER", "Mito", "RNA"]
BLOCKLIST = (  # pycytominer blocklist_features.txt (55 features), regenerated from its pattern
    [f"Nuclei_Correlation_Manders_{a}_{b}" for a, b in permutations(CP_CHANNELS, 2)]
    + [f"Nuclei_Correlation_RWC_{a}_{b}" for a, b in permutations(CP_CHANNELS, 2)]
    + [f"Nuclei_Granularity_{g}_{c}" for g in (14, 15, 16) for c in CP_CHANNELS]
)
assert len(BLOCKLIST) == 55
NAN_FRAC_MAX, OUTLIER_Z, FREQ_CUT, UNIQUE_CUT, CORR_THRESH = 0.01, 500.0, 0.05, 0.01, 0.9


def nan_const_filter(X: np.ndarray, feats: list[str]) -> tuple[np.ndarray, list[str], dict]:
    finite = np.isfinite(X)
    nan_frac = 1 - finite.mean(0)
    X = np.where(finite, X, np.nan)
    std = np.nanstd(X, axis=0)
    drop_nan = nan_frac > NAN_FRAC_MAX
    drop_const = (~drop_nan) & ~(std > 0)
    keep = ~(drop_nan | drop_const)
    return (
        X[:, keep],
        [f for f, k in zip(feats, keep) if k],
        {"drop_nan": int(drop_nan.sum()), "drop_const": int(drop_const.sum())},
    )


def impute_plate_dmso_median(meta: pd.DataFrame, X: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    X = X.copy()
    for _, idx in pd.Series(np.arange(len(meta))).groupby(meta["plate"].to_numpy()).indices.items():
        ref = idx[ref_mask[idx]]
        med = np.nanmedian(X[ref], axis=0) if len(ref) else np.nanmedian(X[idx], axis=0)
        sub = X[idx]
        nan = np.isnan(sub)
        sub[nan] = np.broadcast_to(med, sub.shape)[nan]
        X[idx] = np.nan_to_num(sub, nan=0.0)
    return X


def variance_threshold_drop(Z: np.ndarray) -> np.ndarray:
    drop = np.zeros(Z.shape[1], bool)
    n = Z.shape[0]
    for j in range(Z.shape[1]):
        _, cnt = np.unique(Z[:, j], return_counts=True)
        if len(cnt) < 2:
            drop[j] = True
            continue
        top2 = np.partition(cnt, len(cnt) - 2)[-2:]
        if top2[0] / top2[1] < FREQ_CUT or len(cnt) / n < UNIQUE_CUT:
            drop[j] = True
    return drop


def correlation_prune(Z: np.ndarray, thresh: float = CORR_THRESH) -> np.ndarray:
    R = np.abs(np.corrcoef(Z, rowvar=False))
    R[~np.isfinite(R)] = 0
    order = np.argsort(R.sum(0), kind="stable")
    keep = np.zeros(Z.shape[1], bool)
    kept: list[int] = []
    for j in order:
        if not kept or R[j, kept].max() <= thresh:
            keep[j] = True
            kept.append(j)
    return keep


def select_cp_features(
    Z: np.ndarray, maxabs: np.ndarray, feats: list[str], fit_rows: np.ndarray | None = None
) -> tuple[list[str], dict]:
    """Blocklist, outlier (pre-clip max|z| > 500), variance threshold, greedy |r| > 0.9. Fit on
    ``fit_rows`` (dev-source DMSO, DESIGN.md 4.6) when given."""
    f = np.array(feats)
    F = Z if fit_rows is None else Z[fit_rows]
    m_block = np.isin(f, BLOCKLIST)
    m_out = (~m_block) & (maxabs > OUTLIER_Z)
    keep = ~(m_block | m_out)
    var_drop = np.zeros(len(f), bool)
    var_drop[keep] = variance_threshold_drop(F[:, keep])
    keep &= ~var_drop
    ck = np.zeros(len(f), bool)
    ck[keep] = correlation_prune(F[:, keep])
    info = {
        "drop_blocklist": int(m_block.sum()),
        "drop_outlier": int(m_out.sum()),
        "drop_low_variance": int(var_drop.sum()),
        "drop_correlated": int((keep & ~ck).sum()),
    }
    keep &= ck
    info["n_selected"] = int(keep.sum())
    return f[keep].tolist(), info

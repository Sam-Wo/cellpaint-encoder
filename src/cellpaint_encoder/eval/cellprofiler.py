"""CellProfiler baseline (b-a): per-plate well profiles from the gallery, recomputed on exactly the
evaluation wells (DESIGN.md 2), then the same steps 5-6 as every representation (4.6) after
ShapeShift's feature selection (NaN/constant, blocklist, outlier, variance, |r| > 0.9), fit on
dev-source DMSO.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..data.gallery import Gallery, cp_profile_key
from .profiles import impute_plate_dmso_median, nan_const_filter, robust_z, select_cp_features


def fetch_profiles(g: Gallery, wells: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """``wells``: source, batch, plate, well (evaluation wells). Returns those rows of the plates'
    profile parquets (``Metadata_*`` renamed) with feature columns."""
    plates = wells[["source", "batch", "plate"]].drop_duplicates()

    def one(r):
        d = pd.read_parquet(io.BytesIO(g.get(cp_profile_key(r.source, r.batch, r.plate))))
        keep = set(wells.loc[wells.plate == r.plate, "well"])
        return d[d.Metadata_Well.isin(keep)]

    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(one, plates.itertuples()))
    return pd.concat(frames, ignore_index=True)


def from_profiles(prof: pd.DataFrame, wells_meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Join gallery profiles onto our well metadata (inner join; intersection of features across sources)."""
    feats_by_src = {
        s: [c for c in g.columns if not c.startswith("Metadata_") and g[c].notna().any()]
        for s, g in prof.groupby("Metadata_Source")
    }
    common = sorted(set.intersection(*map(set, feats_by_src.values()))) if feats_by_src else []
    p = prof.rename(columns={"Metadata_Source": "source", "Metadata_Plate": "plate", "Metadata_Well": "well"})
    m = wells_meta.merge(p[["source", "plate", "well", *common]], on=["source", "plate", "well"], how="inner")
    return m.drop(columns=common), m[common].to_numpy(np.float64), common


def prepare_cellprofiler(
    meta: pd.DataFrame,
    X: np.ndarray,
    feats: list[str],
    ref_mask: np.ndarray,
    fit_rows: np.ndarray,
    clip: float = 50.0,
) -> tuple[np.ndarray, list[str], dict]:
    """NaN/constant filter, plate-DMSO-median imputation, robust z, feature selection fit on
    ``fit_rows`` (dev-source DMSO). Returns the selected, z-scored features (whitening follows in
    the shared post-processing)."""
    X, feats, info = nan_const_filter(X, feats)
    X = impute_plate_dmso_median(meta, X, ref_mask)
    Z, zi = robust_z(meta, X, ref_mask, clip=np.inf)
    maxabs = zi["maxabs"]
    Z = np.clip(Z, -clip, clip)
    sel, si = select_cp_features(Z, maxabs, feats, fit_rows)
    keep = np.isin(feats, sel)
    info.update(si)
    return X[:, keep], sel, info

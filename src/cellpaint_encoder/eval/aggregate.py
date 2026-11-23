"""Tile -> site -> well aggregation (DESIGN.md 4.6 steps 3-4), identical for every image model.

Tiles are averaged inside the embedding pass (``embed.run``); here sites are averaged per well.
Well metadata (compound, DMSO flags, row/col, plate split, eval set) is carried along.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["source", "batch", "plate", "well"]


def feature_cols(df: pd.DataFrame, prefix: str = "e") -> list[str]:
    return [c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()]


def sites_to_wells(sites: pd.DataFrame, prefix: str = "e") -> tuple[pd.DataFrame, np.ndarray]:
    feats = feature_cols(sites, prefix)
    if not feats:
        raise ValueError("no embedding columns")
    X = sites[feats].to_numpy(np.float64)
    keys = sites[KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(uniq), X.shape[1]))
    np.add.at(sums, inv, X)
    counts = np.bincount(inv, minlength=len(uniq))
    W = sums / counts[:, None]
    first = pd.Series(np.arange(len(sites))).groupby(inv).first().to_numpy()
    meta_cols = [c for c in sites.columns if c not in feats and c != "site"]
    meta = sites.iloc[first][meta_cols].reset_index(drop=True)
    meta["n_sites"] = counts
    return meta, W.astype(np.float32)

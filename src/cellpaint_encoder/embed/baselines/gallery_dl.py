"""b4: precomputed JUMP gallery DL embeddings (CP-CNN ``cpcnn_zenodo_7114558``, EfficientNetV2-S).

``workspace_dl/embeddings/{model}/{batch}/{plate}/...`` hold per-well-site parquets and
``workspace_dl/profiles/{model}/{batch}/{plate}/`` per-plate well profiles with an ``all_emb``
list column (672-dim float32 for CP-CNN, DESIGN.md 2). We aggregate the per-site files with our
own site->well rule and check against the gallery profile (G-B: r >= 0.99).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ...data.gallery import Gallery

# CP-CNN folder name verified (DESIGN.md 2); the EfficientNet folder name is NOT verified [E]:
# list workspace_dl/profiles/ once in G-B and pass the exact folder name as ``model``.
MODELS = {"cpcnn": "cpcnn_zenodo_7114558"}


def profile_prefix(source: str, model: str, batch: str, plate: str) -> str:
    return f"cpg0016-jump/{source}/workspace_dl/profiles/{MODELS.get(model, model)}/{batch}/{plate}/"


def embedding_prefix(source: str, model: str, batch: str, plate: str) -> str:
    return f"cpg0016-jump/{source}/workspace_dl/embeddings/{MODELS.get(model, model)}/{batch}/{plate}/"


def emb_matrix(df: pd.DataFrame, col: str = "all_emb") -> np.ndarray:
    return np.stack(df[col].to_numpy()).astype(np.float32)


def parse_profile_parquet(data: bytes) -> pd.DataFrame:
    """-> DataFrame with source, plate, well and e0..eD (well level, as shipped)."""
    df = pd.read_parquet(io.BytesIO(data))
    X = emb_matrix(df)
    meta = pd.DataFrame(
        {
            "source": df["Metadata_Source"].astype(str),
            "plate": df["Metadata_Plate"].astype(str),
            "well": df["Metadata_Well"].astype(str),
        }
    )
    return pd.concat([meta, pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])], axis=1)


def parse_site_parquet(data: bytes) -> pd.DataFrame:
    """Per-site embedding file -> source, plate, well, site, e0..eD."""
    df = pd.read_parquet(io.BytesIO(data))
    X = emb_matrix(df)
    cols = {
        "Metadata_Source": "source",
        "Metadata_Plate": "plate",
        "Metadata_Well": "well",
        "Metadata_Site": "site",
    }
    meta = df[[c for c in cols if c in df]].rename(columns=cols).astype(str)
    return pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])], axis=1
    )


def fetch_plate_profiles(g: Gallery, source: str, model: str, batch: str, plate: str) -> pd.DataFrame:
    keys = [
        k
        for k, size in g.list(profile_prefix(source, model, batch, plate))
        if size and k.endswith(".parquet")
    ]
    if not keys:
        raise FileNotFoundError(f"no {model} profile for {source}/{plate}")
    return parse_profile_parquet(g.get(keys[0]))


def fidelity_r(ours_well: np.ndarray, gallery_well: np.ndarray) -> float:
    """Median per-well Pearson r between our site->well aggregate and the gallery profile."""
    a = ours_well - ours_well.mean(1, keepdims=True)
    b = gallery_well - gallery_well.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return float(np.median(r))

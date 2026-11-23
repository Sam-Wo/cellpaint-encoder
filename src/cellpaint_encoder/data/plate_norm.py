"""Compute / load the per-plate DMSO percentiles (``plate_norm.json``) shared by the cache build and
the embedding pass, so a plate is normalised identically for training and evaluation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .gallery import Gallery
from .normalize import plate_percentiles
from .sites import IllumCache, read_site


def compute_plate_norms(m: pd.DataFrame, g: Gallery, il: IllumCache, path: Path, lo: float, hi: float,
                        seed: int, workers: int = 16, plates: set | None = None) -> dict:
    path = Path(path)
    norms = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = m[m.norm & ~m.plate.isin(set(norms))]
    if plates is not None:
        todo = todo[todo.plate.isin(plates)]
    for plate, rows in todo.groupby("plate"):
        with ThreadPoolExecutor(workers) as ex:
            sites = list(ex.map(lambda r: read_site(g, il, r), rows.itertuples()))
        p_lo, p_hi = plate_percentiles(sites, lo, hi, seed=seed)
        norms[plate] = {"source": rows.source.iloc[0], "lo": p_lo.tolist(), "hi": p_hi.tolist(), "n_sites": len(sites)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(norms, indent=1), encoding="utf-8")
    return norms

"""Synthetic well-level embeddings with planted structure, for the evaluation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_wells(
    n_cpd=12,
    sources=("source_2", "source_3", "source_8"),
    plates_per_source=4,
    dim=32,
    cpd_signal=3.0,
    position_signal=0.0,
    source_signal=0.0,
    noise=1.0,
    n_dmso=8,
    seed=0,
    shared_platemap=False,
    shuffle_labels=False,
):
    """Each compound has a fixed direction (``cpd_signal``); each well position adds its own
    direction within a source (``position_signal``, same across plates); each source adds an
    offset (``source_signal``). Compounds sit on 2 plates per source; DMSO wells on every plate."""
    rng = np.random.default_rng(seed)
    cdir = rng.normal(size=(n_cpd, dim))
    pos_dir = {}
    sdir = {s: rng.normal(size=dim) for s in sources}
    rows, X = [], []
    for s in sources:
        plates = [f"{s}_P{i}" for i in range(plates_per_source)]
        for pi, p in enumerate(plates):
            wells_used = set()
            for c in range(n_cpd):
                if pi % 2 != c % 2:
                    continue  # each compound on half the plates of a source
                r, col = (c % 16, (c * 3) % 24) if shared_platemap else (rng.integers(16), rng.integers(24))
                while (r, col) in wells_used:
                    r, col = rng.integers(16), rng.integers(24)
                wells_used.add((r, col))
                rows.append(_row(s, p, r, col, f"CPD{c}", False))
                X.append(
                    _vec(
                        cdir[c] * cpd_signal,
                        s,
                        r,
                        col,
                        pos_dir,
                        sdir,
                        position_signal,
                        source_signal,
                        noise,
                        rng,
                        dim,
                    )
                )
            for _ in range(n_dmso):
                r, col = rng.integers(16), rng.integers(24)
                while (r, col) in wells_used:
                    r, col = rng.integers(16), rng.integers(24)
                wells_used.add((r, col))
                rows.append(_row(s, p, r, col, "DMSO", True))
                X.append(
                    _vec(
                        np.zeros(dim),
                        s,
                        r,
                        col,
                        pos_dir,
                        sdir,
                        position_signal,
                        source_signal,
                        noise,
                        rng,
                        dim,
                    )
                )
    meta = pd.DataFrame(rows)
    if shuffle_labels:
        a = ~meta.is_dmso
        meta.loc[a, "compound_id"] = rng.permutation(meta.loc[a, "compound_id"].to_numpy())
    return meta, np.asarray(X, np.float32)


def _row(s, p, r, c, cpd, dmso):
    return {
        "source": s,
        "batch": f"{s}_B{p[-1]}",
        "plate": p,
        "well": f"{chr(65 + r)}{c + 1:02d}",
        "row": int(r),
        "col": int(c),
        "compound_id": cpd,
        "is_dmso": dmso,
        "is_anchor": not dmso,
        "eval_dmso": dmso,
        "plate_split": "train",
        "eval_set": "E1",
    }


def _vec(base, s, r, c, pos_dir, sdir, ps, ss, noise, rng, dim):
    key = (s, r, c)
    if key not in pos_dir:
        pos_dir[key] = rng.normal(size=dim)
    edge = 1.0 if (r in (0, 15) or c in (0, 23)) else 0.0
    return base + ps * (pos_dir[key] + edge * 2.0) + ss * sdir[s] + noise * rng.normal(size=dim)

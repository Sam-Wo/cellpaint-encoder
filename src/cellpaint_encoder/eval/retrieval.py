"""copairs-style retrieval metrics (DESIGN.md 4.2-4.3).

Implemented here rather than by calling ``copairs`` so that every pair mask of the
pre-registration is explicit and unit-tested. Same definitions as copairs: each query ranks its
candidate set by cosine similarity; AP is the mean of precision@k over the ranks k of the
positives; a group's mAP is the mean AP of its queries; the null of a query is the AP of a random
ranking with the same number of positives and candidates; a group's p-value compares its mAP
with the mean of its queries' null APs (``n_null`` draws), ``p = (1 + #null >= obs) / (1 + n)``;
Benjamini-Hochberg across groups.

* **Primary** (4.2) :func:`cross_source`: queries = anchor-compound wells of the query source;
  candidates = anchor-compound wells of the reference sources (never the query's own source);
  positives = same compound, negatives = other compounds; ``strict`` also drops candidates at the
  query's well ID. Score: mAP per compound, averaged over compounds.
* **M1 / M2** :func:`within_source`: candidates in the same source; M1 restricts the whole
  candidate pool to other plates and other well positions (``position_restricted=True``); M2 only
  removes the query itself. Units are source x compound.
* **M3 / M4** :func:`label_retrieval` on compound consensus profiles: M3 single-label (MoA);
  M4 target annotation ``multilabel=True`` (positives share >= 1 target), with a label-shuffle null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1)


def average_precision(scores: np.ndarray, pos: np.ndarray) -> float:
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    p = pos[order]
    ranks = np.flatnonzero(p) + 1
    return float((np.arange(1, len(ranks) + 1) / ranks).mean())


def null_ap(n_pos: int, n_total: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    if n_pos == 0 or n_total == 0:
        return np.full(n_draws, np.nan)
    ranks = np.sort(np.argsort(rng.random((n_draws, n_total)), axis=1)[:, :n_pos] + 1, axis=1)
    return (np.arange(1, n_pos + 1)[None, :] / ranks).mean(1)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def _queries(
    meta: pd.DataFrame, U: np.ndarray, qidx: np.ndarray, cand_fn, labels: np.ndarray, pos_fn=None
) -> pd.DataFrame:
    """One row per query. Positives: same ``labels`` value, or ``pos_fn(i, c)`` (multilabel)."""
    rows = []
    for i in qidx:
        cand = cand_fn(i)
        cand[i] = False
        c = np.flatnonzero(cand)
        pos = labels[c] == labels[i] if pos_fn is None else pos_fn(i, c)
        s = U[c] @ U[i]
        rows.append(
            {
                "query": int(i),
                "label": labels[i],
                "n_pos": int(pos.sum()),
                "n_total": len(c),
                "ap": average_precision(s, pos) if len(c) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=["query", "label", "n_pos", "n_total", "ap"])


def summarise(
    q: pd.DataFrame, group_cols: list[str], n_null: int = 1000, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Per-group mAP with permutation p-values and BH; overall mean over groups with a positive."""
    q = q[q.n_pos > 0]
    if q.empty:
        return pd.DataFrame(columns=[*group_cols, "mAP", "n_queries", "p", "q"]), {
            "mAP": float("nan"),
            "n_groups": 0,
            "frac_q05": float("nan"),
            "note": "no query with a positive",
        }
    rng = np.random.default_rng(seed)
    cache: dict[tuple[int, int], np.ndarray] = {}
    out = []
    for key, g in q.groupby(group_cols, sort=True):
        nulls = []
        for r in g.itertuples():
            k = (r.n_pos, r.n_total)
            if k not in cache:
                cache[k] = null_ap(r.n_pos, r.n_total, n_null, rng)
            nulls.append(cache[k])
        null = np.mean(nulls, axis=0)
        obs = g.ap.mean()
        p = (1 + np.sum(null >= obs - 1e-12)) / (1 + n_null)
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(group_cols, key)), "mAP": obs, "n_queries": len(g), "p": p})
    df = pd.DataFrame(out)
    df["q"] = bh(df.p.to_numpy())
    return df, {
        "mAP": float(df.mAP.mean()),
        "n_groups": len(df),
        "frac_q05": float((df.q < 0.05).mean()),
        "n_queries": int(df.n_queries.sum()),
    }


def _arr(meta, col):
    return meta[col].astype(str).to_numpy()


def cross_source(
    meta: pd.DataFrame,
    X: np.ndarray,
    query_sources,
    ref_sources,
    strict: bool = False,
    n_null: int = 1000,
    seed: int = 0,
    query_mask: np.ndarray | None = None,
):
    """Primary metric (4.2). ``query_mask`` further restricts queries (e.g. held-out plates, M6)."""
    U = unit_rows(X.astype(np.float64))
    src, well, cpd = _arr(meta, "source"), _arr(meta, "well"), _arr(meta, "compound_id")
    anchor = meta["is_anchor"].to_numpy().astype(bool)
    ref = anchor & np.isin(src, list(ref_sources))
    qm = anchor & np.isin(src, list(query_sources))
    if query_mask is not None:
        qm &= query_mask

    def cand(i):
        c = ref & (src != src[i])
        if strict:
            c &= well != well[i]
        return c.copy()

    q = _queries(meta, U, np.flatnonzero(qm), cand, cpd)
    q["compound_id"] = q.label
    per, summ = summarise(q, ["compound_id"], n_null, seed)
    summ.update({"n_query_wells": int(qm.sum()), "n_reference_wells": int(ref.sum()), "strict": strict})
    return per, summ, q


def within_source(
    meta: pd.DataFrame,
    X: np.ndarray,
    sources,
    position_restricted: bool = True,
    dmso_negatives: bool = False,
    n_null: int = 1000,
    seed: int = 0,
):
    """M1 (``position_restricted``) / M2. With ``dmso_negatives`` the negatives are the source's
    DMSO wells instead of other compounds (continuous mAP only, no binary calls)."""
    U = unit_rows(X.astype(np.float64))
    src, well, plate, cpd = (_arr(meta, c) for c in ("source", "well", "plate", "compound_id"))
    anchor = meta["is_anchor"].to_numpy().astype(bool)
    dmso = meta["is_dmso"].to_numpy().astype(bool)
    qm = anchor & np.isin(src, list(sources))

    def cand(i):
        base = src == src[i]
        if position_restricted:
            base = base & (plate != plate[i]) & (well != well[i])
        if dmso_negatives:
            return base & ((anchor & (cpd == cpd[i])) | dmso)
        return (base & anchor).copy()

    q = _queries(meta, U, np.flatnonzero(qm), cand, cpd)
    q["source"] = src[q["query"].to_numpy()] if len(q) else []
    q["compound_id"] = q.label
    per, summ = summarise(q, ["source", "compound_id"], n_null, seed)
    summ.update({"position_restricted": position_restricted, "dmso_negatives": dmso_negatives})
    return per, summ, q


def consensus(meta: pd.DataFrame, X: np.ndarray, sources=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Compound consensus: median of wells within each source, then mean across sources."""
    m = meta["is_anchor"].to_numpy().astype(bool)
    if sources is not None:
        m &= meta["source"].isin(list(sources)).to_numpy()
    d = meta[m]
    Xm = X[m]
    per_src = {}
    for (c, _src), idx in (
        pd.Series(np.arange(len(d)))
        .groupby([d["compound_id"].to_numpy(), d["source"].to_numpy()])
        .indices.items()
    ):
        per_src.setdefault(c, []).append(np.median(Xm[idx], axis=0))
    cpds = sorted(per_src)
    C = np.stack([np.mean(per_src[c], axis=0) for c in cpds]) if cpds else np.zeros((0, X.shape[1]))
    return pd.DataFrame({"compound_id": cpds, "n_sources": [len(per_src[c]) for c in cpds]}), C


def label_retrieval(
    labels,
    X: np.ndarray,
    min_members: int = 3,
    n_null: int = 1000,
    seed: int = 0,
    multilabel: bool = False,
    n_shuffle: int = 0,
):
    """M3/M4: each annotated compound queries all other annotated compounds; positives share a label.

    Single-label (M3, ``multilabel=False``): only classes with >= ``min_members`` members are
    queried (all annotated compounds stay candidates); the headline ``mAP`` is the mean over classes
    of the class mean AP.

    Multilabel (M4 target annotation): ``labels[i]`` is a set of labels (or a ``|``-separated
    string); a compound is annotated if it has >= 1 label. Only labels held by >= ``min_members``
    annotated compounds count; a query is an annotated compound holding >= 1 such label, its
    positives are the other annotated compounds sharing >= 1 such label, its negatives are all other
    annotated compounds (annotated compounds without a counted label are candidates, never queries).
    The headline ``mAP`` is the mean AP over queries (each query is its own group, with its own
    permutation p-value).

    ``n_shuffle`` > 0 adds a label-shuffle null: the annotated compounds' label (sets) are permuted
    among them ``n_shuffle`` times (seeded by ``seed`` only, so every representation sees the same
    permutations) and the headline statistic is recomputed on the same profiles."""
    if multilabel:
        return _multilabel_retrieval(labels, X, min_members, n_null, seed, n_shuffle)
    labels = np.asarray(labels, dtype=object)
    ok = pd.notna(labels) & (labels != "") & (labels != "unclear")
    idx = np.flatnonzero(ok)
    vc = pd.Series(labels[idx]).value_counts()
    big = set(vc[vc >= min_members].index)
    U = unit_rows(X[idx].astype(np.float64))
    lab = labels[idx]
    meta = pd.DataFrame({"label": lab})
    qidx = np.flatnonzero(np.isin(lab, list(big)))

    def cand(i):
        return np.ones(len(idx), bool)

    q = _queries(meta, U, qidx, cand, lab.astype(str))
    per, summ = summarise(q, ["label"], n_null, seed)
    if len(q):
        q["query"] = idx[q["query"].to_numpy()]  # back to the caller's row index
    summ["n_classes"] = len(big)
    summ["mean_ap_over_queries"] = float(q.ap.mean()) if len(q) else float("nan")
    if n_shuffle:
        classes = sorted(big, key=str)
        L = np.stack([lab == c for c in classes], axis=1) if classes else np.zeros((len(lab), 0), bool)
        summ.update(_shuffle_null(U, L, n_shuffle, seed, summ["mAP"], per_class=True))
    return per, summ, q


def _membership(labels, min_members: int):
    """Annotated rows, their label sets, the counted labels (>= min_members) and the bool matrix."""
    from .targets import parse_targets

    sets = [frozenset(t for t in parse_targets(v) if t != "unclear") for v in labels]
    idx = np.array([i for i, s in enumerate(sets) if s], dtype=int)
    counts = pd.Series([t for i in idx for t in sets[i]], dtype=object).value_counts()
    big = sorted(counts[counts >= min_members].index)
    col = {t: j for j, t in enumerate(big)}
    L = np.zeros((len(idx), len(big)), bool)
    for r, i in enumerate(idx):
        for t in sets[i]:
            if t in col:
                L[r, col[t]] = True
    return idx, [sets[i] for i in idx], big, L


def _multilabel_retrieval(labels, X, min_members, n_null, seed, n_shuffle):
    idx, sets, big, L = _membership(list(labels), min_members)
    U = unit_rows(X[idx].astype(np.float64)) if len(idx) else np.zeros((0, X.shape[1]))
    bigset = set(big)
    shown = np.array(["|".join(sorted(s & bigset)) for s in sets], dtype=object)
    qidx = np.flatnonzero(L.any(1)) if len(idx) else np.zeros(0, int)

    def cand(i):
        return np.ones(len(idx), bool)

    def pos(i, c):
        return (L[c] & L[i]).any(1)

    q = _queries(pd.DataFrame({"label": shown}), U, qidx, cand, shown, pos_fn=pos)
    per, summ = summarise(q, ["query"], n_null, seed)
    if len(q):
        q["query"] = idx[q["query"].to_numpy()]  # back to the caller's row index
    if len(per):
        per["query"] = idx[per["query"].to_numpy().astype(int)]
        per["label"] = shown[np.searchsorted(idx, per["query"].to_numpy())]
    summ["n_classes"] = summ["n_targets"] = len(big)
    summ["n_annotated"] = len(idx)
    summ["mean_ap_over_queries"] = float(q.ap.mean()) if len(q) else float("nan")
    if n_shuffle:
        summ.update(_shuffle_null(U, L, n_shuffle, seed, summ["mAP"], per_class=False))
    return per, summ, q


def _headline(S: np.ndarray, L: np.ndarray, per_class: bool) -> float:
    """Headline mAP for membership ``L`` on the similarity matrix ``S`` (same candidate order and
    tie-breaking as :func:`_queries`): mean over queries, or mean over classes of class means."""
    n = len(S)
    aps, cls = [], []
    for i in np.flatnonzero(L.any(1)):
        c = np.r_[0:i, i + 1 : n]
        p = (L[c] & L[i]).any(1)
        if p.any():
            aps.append(average_precision(S[i, c], p))
            cls.append(int(np.argmax(L[i])))
    if not aps:
        return float("nan")
    if per_class:
        return float(pd.Series(aps).groupby(cls).mean().mean())
    return float(np.mean(aps))


def _shuffle_null(U, L, n_shuffle: int, seed: int, observed: float, per_class: bool) -> dict:
    rng = np.random.default_rng([seed, 7919])  # independent of the AP-null stream, same for every rep
    S = U @ U.T
    null = np.array([_headline(S, L[rng.permutation(len(L))], per_class) for _ in range(n_shuffle)])
    ok = null[np.isfinite(null)]
    return {
        "n_shuffle": int(n_shuffle),
        "shuffle_null_mean": float(ok.mean()) if len(ok) else float("nan"),
        "shuffle_null_sd": float(ok.std(ddof=1)) if len(ok) > 1 else float("nan"),
        "shuffle_null_q95": float(np.quantile(ok, 0.95)) if len(ok) else float("nan"),
        "p_shuffle": float((1 + np.sum(ok >= observed - 1e-12)) / (1 + len(ok)))
        if len(ok) and np.isfinite(observed)
        else float("nan"),
    }

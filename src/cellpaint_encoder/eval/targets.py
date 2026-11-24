"""Target annotations for E3 (Target-2) compounds, for the target-annotation half of M4 (DESIGN.md 4.3).

Sources (pinned; fetched by ``scripts/fetch_jump_metadata.py``):

* ``JUMP-Target-2_compound_metadata.tsv`` from jump-cellpainting/JUMP-Target (MIT): one row per
  Broad sample, with ``InChIKey``, ``pert_iname``, ``target_list`` (pipe-separated gene symbols),
  ``pert_type`` (``trt`` | ``control``) and ``control_type`` (``poscon_*`` | ``negcon`` | NA).
* ``metadata/compound.csv.gz`` from jump-cellpainting/datasets (BSD-3-Clause):
  ``Metadata_JCP2022`` -> ``Metadata_InChIKey``.

E3 manifest rows carry ``compound_id`` = ``Metadata_JCP2022`` from JUMP ``well.csv.gz`` (see
``scripts/build_manifest.py``), so the join is ``compound_id -> InChIKey -> target_list``.

Rules (:func:`build_target_table`):

1. Target-2 rows with ``pert_type != "trt"`` or a non-missing ``control_type`` are controls
   (positive controls and the DMSO negative control). An InChIKey is a control if any of its rows is.
2. Target-2 rows are collapsed per full InChIKey; the target set of a key is the **union** of the
   ``target_list`` of its rows (e.g. quinine and quinidine share one standard InChIKey).
3. Each E3 compound (DMSO ``JCP2022_033924`` excluded) is joined on the **full** InChIKey. Only an
   E3 compound left unmatched falls back to the first 14 characters (the connectivity block), and
   only when that block identifies exactly one Target-2 InChIKey; the rule used is recorded per
   compound (``join_rule``) and counted in the report.
4. The table keeps the treatment compounds that map and have >= 1 target.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pandas as pd

DMSO_JCP2022 = "JCP2022_033924"


def parse_targets(v) -> frozenset[str]:
    """Pipe-separated target string (or an iterable of targets) -> set; missing/empty -> empty set."""
    if isinstance(v, str):
        parts = v.split("|")
    elif isinstance(v, (set, frozenset, list, tuple)):
        parts = list(v)
    elif v is None or pd.isna(v):
        return frozenset()
    else:
        raise TypeError(f"cannot read targets from {v!r}")
    return frozenset(p.strip() for p in parts if isinstance(p, str) and p.strip())


def load_target2(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)


def collapse_target2(t2: pd.DataFrame) -> pd.DataFrame:
    """One row per full InChIKey: is_control, pert_iname (|-joined), targets (union), n_rows."""
    t = t2.dropna(subset=["InChIKey"]).copy()
    ctrl = (t["pert_type"].fillna("") != "trt") | t["control_type"].notna()
    t["_ctrl"] = ctrl.to_numpy()
    rows = []
    for key, g in t.groupby("InChIKey", sort=True):
        tg = frozenset().union(*(parse_targets(v) for v in g["target_list"]))
        rows.append(
            {
                "inchikey": key,
                "is_control": bool(g["_ctrl"].any()),
                "pert_iname": "|".join(sorted(set(g["pert_iname"].dropna()))),
                "targets": tg,
                "n_rows": len(g),
            }
        )
    return pd.DataFrame(rows, columns=["inchikey", "is_control", "pert_iname", "targets", "n_rows"])


def build_target_table(
    t2: pd.DataFrame, compounds: pd.DataFrame, compound_ids: Iterable[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Map E3 ``compound_id`` (JCP2022) -> InChIKey -> Target-2 targets.

    ``compounds``: JUMP ``compound.csv.gz`` (``Metadata_JCP2022``, ``Metadata_InChIKey``).
    ``compound_ids``: the E3 compound IDs (default: every JCP2022 ID whose InChIKey is in Target-2).
    Returns (table, report); the table has ``compound_id, inchikey, pert_iname, targets, join_rule``
    with ``targets`` pipe-joined and sorted."""
    col = collapse_target2(t2)
    by_key = col.set_index("inchikey")
    block: dict[str, list[str]] = {}
    for k in col.inchikey:
        block.setdefault(k[:14], []).append(k)
    cmap = compounds.dropna(subset=["Metadata_JCP2022"]).drop_duplicates("Metadata_JCP2022")
    cmap = dict(zip(cmap["Metadata_JCP2022"].astype(str), cmap["Metadata_InChIKey"]))
    if compound_ids is None:
        keys14 = set(block)
        ids = sorted(
            j for j, k in cmap.items() if isinstance(k, str) and (k in by_key.index or k[:14] in keys14)
        )
    else:
        ids = sorted({str(c) for c in compound_ids if isinstance(c, str) and c})
    n_dmso = int(DMSO_JCP2022 in ids)
    ids = [c for c in ids if c != DMSO_JCP2022]
    rows, cnt = [], Counter()
    for cid in ids:
        key = cmap.get(cid)
        if not isinstance(key, str):
            cnt["not_in_compound_csv"] += 1
            continue
        if key in by_key.index:
            hit, rule = key, "full"
        elif len(block.get(key[:14], [])) == 1:
            hit, rule = block[key[:14]][0], "connectivity14"
        else:
            cnt["unmapped_ambiguous14" if key[:14] in block else "unmapped"] += 1
            continue
        cnt[f"mapped_{rule}"] += 1
        r = by_key.loc[hit]
        if r.is_control:
            cnt["control"] += 1
            continue
        if not r.targets:
            cnt["treatment_no_targets"] += 1
            continue
        rows.append(
            {
                "compound_id": cid,
                "inchikey": hit,
                "pert_iname": r.pert_iname,
                "targets": "|".join(sorted(r.targets)),
                "join_rule": rule,
            }
        )
    table = pd.DataFrame(rows, columns=["compound_id", "inchikey", "pert_iname", "targets", "join_rule"])
    trt = col[~col.is_control]
    mapped_keys = set(table.inchikey)
    report = {
        "target2_rows": len(t2),
        "target2_control_rows": int(
            ((t2["pert_type"].fillna("") != "trt") | t2["control_type"].notna()).sum()
        ),
        "target2_inchikeys": len(col),
        "target2_treatment_inchikeys": len(trt),
        "target2_control_inchikeys": int(col.is_control.sum()),
        "target2_inchikeys_with_several_rows": int((col.n_rows > 1).sum()),
        "e3_compounds_excl_dmso": len(ids),
        "e3_dmso_present": bool(n_dmso),
        "mapped_full": cnt["mapped_full"],
        "mapped_connectivity14": cnt["mapped_connectivity14"],
        "unmapped": cnt["unmapped"] + cnt["unmapped_ambiguous14"] + cnt["not_in_compound_csv"],
        "unmapped_detail": {
            k: cnt[k] for k in ("unmapped", "unmapped_ambiguous14", "not_in_compound_csv") if cnt[k]
        },
        "excluded_controls": cnt["control"],
        "treatment_without_targets": cnt["treatment_no_targets"],
        "annotated_treatments": len(table),
        "target2_treatments_not_in_e3": int(len(set(trt.inchikey) - mapped_keys)),
        "join_rule": "full" if cnt["mapped_connectivity14"] == 0 else "full, then connectivity14 fallback",
    }
    report.update(annotation_counts(table.targets))
    return table, report


def annotation_counts(target_strings: Iterable, min_members: int = 3) -> dict:
    """Counts behind the M4 target-annotation metric: targets with >= ``min_members`` annotated
    compounds, and the compounds holding at least one such target (the queries)."""
    sets = [parse_targets(v) for v in target_strings]
    sets = [s for s in sets if s]
    c = Counter(t for s in sets for t in s)
    big = {t for t, n in c.items() if n >= min_members}
    return {
        "annotated_compounds": len(sets),
        "n_targets_any": len(c),
        f"n_targets_ge{min_members}": len(big),
        f"n_queries_ge{min_members}": sum(bool(s & big) for s in sets),
    }

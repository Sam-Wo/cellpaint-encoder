"""Build the site-level split manifest (DESIGN.md 1.2, 4.1).

One row per site with role flags:

* ``train``  - site goes into the training tile cache. Only plates of the training sources that
  are not held out, and never an anchor-compound well or one of the plate's evaluation DMSO wells.
* ``eval_set`` - E1 (anchor wells + eval DMSO of the evaluation sources), E2 (source_7, low dose),
  E3 (Target-2 plates), E4 (all-DMSO plates), or "" for none.
* ``dev``    - dev cache: E1 rows of the dev sources, anchor wells + the first 4 of the 16 DMSO wells.
* ``norm``   - the DMSO sites used for the per-plate intensity percentiles (subset of eval DMSO).

Everything random is a function of ``data.manifest_seed`` and the plate name, so the manifest
is identical for every training seed. :func:`check_manifest` asserts the leakage rules that
``tests/test_splits.py`` also checks.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import DataCfg

ROLE_COLS = ["train", "eval_set", "dev", "norm", "eval_dmso", "dev_dmso"]


def _rng(seed: int, *parts: str) -> np.random.Generator:
    h = hashlib.sha256("|".join([str(seed), *parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def split_plates(plates: pd.DataFrame, frac: float, seed: int) -> dict[str, str]:
    """Per source, hold out ``round(frac * n)`` plates, stratified by batch: plates are shuffled
    within batch, batches are interleaved, and the first plates of that order are held out."""
    out = {}
    for src, g in plates.groupby("source"):
        per_batch = []
        for b, gb in sorted(g.groupby("batch"), key=lambda t: t[0]):
            p = sorted(gb.plate.unique())
            _rng(seed, "split", src, b).shuffle(p)
            per_batch.append(p)
        order = []
        while any(per_batch):
            for lst in per_batch:
                if lst:
                    order.append(lst.pop(0))
        n_hold = int(round(frac * len(order)))
        for i, p in enumerate(order):
            out[p] = "heldout" if i < n_hold else "train"
    return out


def build_manifest(cfg: DataCfg, anchor: pd.DataFrame, load_data: dict[str, pd.DataFrame],
                   anchor_low: pd.DataFrame | None = None,
                   extra_plates: pd.DataFrame | None = None) -> pd.DataFrame:
    """``anchor``: source,batch,plate,well,compound_id,is_dmso (10 uM, all anchor + DMSO wells of the
    anchor plates). ``load_data``: plate -> parsed load_data (see load_data.parse_load_data).
    ``extra_plates``: optional rows source,batch,plate,eval_set (E3/E4) with all wells of those
    plates in ``load_data`` and compound_id/is_dmso joined from JUMP well.csv when available."""
    seed = cfg.manifest_seed
    anchor = anchor.copy()
    anchor["is_dmso"] = anchor["is_dmso"].astype(bool)
    excluded = set(cfg.excluded_sources)
    use_sources = (set(cfg.train_sources) | set(cfg.eval_sources)) - excluded
    if cfg.heldout_source in cfg.train_sources:
        raise ValueError("held-out source listed as a training source")
    a = anchor[anchor.source.isin(use_sources)]
    if cfg.plates:
        a = a[a.plate.isin(set(cfg.plates))]
    plates = a[["source", "batch", "plate"]].drop_duplicates().sort_values(["source", "plate"])
    train_plates = plates[plates.source.isin(cfg.train_sources)]
    split = split_plates(train_plates, cfg.heldout_plate_frac, seed)

    frames = []
    for r in plates.itertuples():
        if r.plate not in load_data:
            raise KeyError(f"missing load_data for plate {r.plate}")
        ld = load_data[r.plate]
        if (ld.source != r.source).any():
            raise ValueError(f"load_data source mismatch for {r.plate}")
        aw = a[a.plate == r.plate].set_index("well")
        frames.append(_plate_rows(cfg, ld, aw, r.source, split.get(r.plate, "eval_only"), "E1", seed))
    if anchor_low is not None:
        lo = anchor_low[~anchor_low.source.isin(set(cfg.train_sources) | {cfg.heldout_source})]
        for r in lo[["source", "batch", "plate"]].drop_duplicates().itertuples():
            if r.plate in load_data:
                aw = lo[lo.plate == r.plate].set_index("well")
                frames.append(_plate_rows(cfg, load_data[r.plate], aw, r.source, "eval_only", "E2", seed))
    if extra_plates is not None:
        for r in extra_plates.itertuples():
            frames.append(_extra_rows(cfg, load_data[r.plate], r.eval_set, seed))
    m = pd.concat(frames, ignore_index=True)
    m = m[m.train | (m.eval_set != "") | m.norm].reset_index(drop=True)
    check_manifest(m, cfg)
    return m


def _plate_rows(cfg: DataCfg, ld: pd.DataFrame, aw: pd.DataFrame, source: str, plate_split: str,
                eval_set: str, seed: int) -> pd.DataFrame:
    plate = ld.plate.iloc[0]
    d = ld.copy()
    d["compound_id"] = d.well.map(aw["compound_id"]).fillna("")
    d["is_dmso"] = d.well.map(aw["is_dmso"].astype(bool).to_dict()).eq(True)
    d["is_anchor"] = (d.compound_id != "") & ~d.is_dmso  # all anchor compounds, before any restriction
    d["plate_split"] = plate_split
    for c in ROLE_COLS:
        d[c] = "" if c == "eval_set" else False

    # 16 evaluation DMSO wells (masked from training); dev = first 4 of them
    dmso_wells = sorted(d.loc[d.is_dmso, "well"].unique())
    rng = _rng(seed, "dmso", plate)
    rng.shuffle(dmso_wells)
    eval_dmso = dmso_wells[: cfg.n_eval_dmso_per_plate]
    dev_dmso = set(eval_dmso[: cfg.n_dev_dmso_per_plate])
    d["eval_dmso"] = d.well.isin(set(eval_dmso))
    d["dev_dmso"] = d.well.isin(dev_dmso)

    # evaluation rows
    anchor_wells = sorted(d.loc[d.is_anchor, "well"].unique())
    if cfg.eval_compounds:
        keep = set(cfg.eval_compounds)
        anchor_wells = [w for w in anchor_wells if d.loc[d.well == w, "compound_id"].iloc[0] in keep]
    if cfg.max_anchor_wells_per_plate:
        anchor_wells = anchor_wells[: cfg.max_anchor_wells_per_plate]
    ev_wells = set(anchor_wells) | set(eval_dmso)
    ev = d.well.isin(ev_wells)
    if cfg.eval_sites_per_well:
        ev &= _first_sites(d, cfg.eval_sites_per_well, _rng(seed, "evsites", plate))
    if source in cfg.eval_sources or eval_set == "E2":
        d.loc[ev, "eval_set"] = eval_set
        if eval_set == "E1" and source in cfg.train_sources and source != cfg.heldout_source:
            d.loc[ev & (d.is_anchor | d.dev_dmso), "dev"] = True
    # per-plate normalisation sites: drawn from the evaluation DMSO sites (downloaded anyway for E1)
    cand = d.index[d.eval_dmso & ev].to_numpy()
    if len(cand):
        take = _rng(seed, "norm", plate).permutation(cand)[: cfg.norm_sites_per_plate]
        d.loc[take, "norm"] = True

    # training rows: unmasked wells on training plates, sites_per_well random sites each
    if plate_split == "train" and eval_set == "E1":
        masked = d.is_anchor | d.eval_dmso
        wells = sorted(d.loc[~masked, "well"].unique())
        rng = _rng(seed, "train", plate)
        if cfg.max_train_wells_per_plate:
            wells = list(rng.permutation(wells)[: cfg.max_train_wells_per_plate])
        tr = d.well.isin(set(wells)) & ~masked
        tr &= _first_sites(d, cfg.sites_per_well, _rng(seed, "sites", plate))
        d.loc[tr, "train"] = True
    return d


def _first_sites(d: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.Series:
    """Boolean mask choosing ``k`` random sites per well (all sites if a well has fewer)."""
    keep = pd.Series(False, index=d.index)
    for _, idx in d.groupby("well").groups.items():
        idx = np.asarray(sorted(idx))
        keep.loc[rng.permutation(idx)[:k]] = True
    return keep


def _extra_rows(cfg: DataCfg, ld: pd.DataFrame, eval_set: str, seed: int) -> pd.DataFrame:
    d = ld.copy()
    for c, v in (("compound_id", ""), ("is_dmso", False), ("is_anchor", False)):
        if c not in d:
            d[c] = v
    d["plate_split"] = "eval_only"
    for c in ROLE_COLS:
        d[c] = "" if c == "eval_set" else False
    keep = _first_sites(d, cfg.extra_sites_per_well, _rng(seed, "extra", d.plate.iloc[0]))
    d.loc[keep, "eval_set"] = eval_set
    if eval_set == "E4":
        d["is_dmso"] = True
    return d


def extra_eval_plates(cfg: DataCfg, plate_table: pd.DataFrame) -> pd.DataFrame:
    """E3 (TARGET2) and E4 (DMSO) plate choice from JUMP ``plate.csv.gz``, seeded per source."""
    pt = plate_table.rename(columns={"Metadata_Source": "source", "Metadata_Batch": "batch",
                                     "Metadata_Plate": "plate", "Metadata_PlateType": "type"})
    rows = []
    for eval_set, ptype, sources, n in (("E3", "TARGET2", cfg.e3_sources, cfg.e3_plates_per_source),
                                        ("E4", "DMSO", cfg.e4_sources, cfg.e4_plates_per_source)):
        for s in sources:
            p = sorted(pt[(pt.source == s) & (pt.type == ptype)].plate)
            p = list(_rng(cfg.manifest_seed, eval_set, s).permutation(p)[:n]) if p else []
            for plate in p:
                b = pt[(pt.source == s) & (pt.plate == plate)].batch.iloc[0]
                rows.append({"source": s, "batch": b, "plate": plate, "eval_set": eval_set})
    return pd.DataFrame(rows, columns=["source", "batch", "plate", "eval_set"])


def check_manifest(m: pd.DataFrame, cfg: DataCfg) -> None:
    tr = m[m.train]
    if cfg.heldout_source in set(tr.source):
        raise AssertionError(f"{cfg.heldout_source} present in the training manifest")
    if tr.is_anchor.any():
        raise AssertionError("anchor-compound well in the training manifest")
    if tr.eval_dmso.any():
        raise AssertionError("evaluation DMSO well in the training manifest")
    if (tr.plate_split != "train").any():
        raise AssertionError("held-out plate in the training manifest")
    if not set(tr.source) <= set(cfg.train_sources):
        raise AssertionError("training rows from a non-training source")
    if (m.dev & (m.source == cfg.heldout_source)).any():
        raise AssertionError("held-out source in the dev cache")
    if (m.dev & (m.plate_split == "heldout")).any():
        pass  # allowed: dev uses anchor wells on any training-source plate (S-plate diagnostics use them)
    if m.duplicated(["plate", "well", "site"]).any():
        raise AssertionError("duplicate sites in the manifest")

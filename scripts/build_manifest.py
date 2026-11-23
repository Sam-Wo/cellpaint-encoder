"""Fetch load_data.csv for every needed plate and write the split manifest (DESIGN.md 1.2).

py -3.13 -I -X utf8 scripts/build_manifest.py --config configs/full.yaml [key=value ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.gallery import Gallery, load_data_key
from cellpaint_encoder.data.load_data import parse_load_data
from cellpaint_encoder.data.manifest import build_manifest, extra_eval_plates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dc = cfg.data
    g = Gallery(mirror=dc.raw_mirror, byte_cap=dc.byte_cap)
    anchor = pd.read_parquet(dc.anchor_wells)
    low = pd.read_parquet(dc.anchor_wells_low) if dc.anchor_wells_low else None
    use = (set(dc.train_sources) | set(dc.eval_sources)) - set(dc.excluded_sources)
    need = anchor[anchor.source.isin(use)]
    if dc.plates:
        need = need[need.plate.isin(set(dc.plates))]
    plates = need[["source", "batch", "plate"]].drop_duplicates()
    if low is not None:
        lo = low[~low.source.isin(set(dc.train_sources) | {dc.heldout_source})]
        if dc.plates:
            lo = lo[lo.plate.isin(set(dc.plates))]
        plates = pd.concat([plates, lo[["source", "batch", "plate"]].drop_duplicates()])
    extra = None
    if dc.plate_table:
        extra = extra_eval_plates(dc, pd.read_csv(dc.plate_table))
        plates = pd.concat([plates, extra[["source", "batch", "plate"]]])
    plates = plates.drop_duplicates("plate")
    print(f"plates: {len(plates)}")

    def fetch(r):
        return r.plate, parse_load_data(g.get(load_data_key(r.source, r.batch, r.plate)))

    with ThreadPoolExecutor(16) as ex:
        ld = dict(ex.map(fetch, plates.itertuples()))
    if extra is not None and dc.jump_wells:
        wells = pd.read_csv(dc.jump_wells).rename(
            columns={"Metadata_Plate": "plate", "Metadata_Well": "well", "Metadata_JCP2022": "compound_id"}
        )
        for p in extra.plate:
            w = wells[wells.plate == p].set_index("well").compound_id
            ld[p] = ld[p].assign(compound_id=ld[p].well.map(w).fillna(""))
            ld[p]["is_dmso"] = ld[p].compound_id == "JCP2022_033924"
    m = build_manifest(dc, anchor, ld, anchor_low=low, extra_plates=extra)
    out = Path(dc.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    summary = {
        "sites": len(m),
        "train_sites": int(m.train.sum()),
        "dev_sites": int(m.dev.sum()),
        "norm_sites": int(m.norm.sum()),
        "eval_sites": m[m.eval_set != ""].groupby("eval_set").size().to_dict(),
        "train_wells": int(m[m.train].groupby(["plate", "well"]).ngroups),
        "train_plates_by_source": m[m.train].groupby("source").plate.nunique().to_dict(),
        "heldout_plates_by_source": m[m.plate_split == "heldout"].groupby("source").plate.nunique().to_dict(),
        "bytes_downloaded": g.bytes_downloaded,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

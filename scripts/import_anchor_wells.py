"""Copy the ShapeShift anchor-well metadata (no features) and MoA labels into data/anchor/.

    py -3.13 -I -X utf8 scripts/import_anchor_wells.py --shapeshift-step2 <path-to-shapeshift/step2>

Writes ``anchor_wells_high.parquet``, ``anchor_wells_low.parquet`` (source, batch, plate, well,
compound_id, is_dmso) and ``moa.csv`` (compound_id, moa; Tahoe moa-fine, 'unclear' dropped).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
from pathlib import Path

import pandas as pd

COLS = ["source", "batch", "plate", "well", "compound_id", "is_dmso"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapeshift-step2", required=True)
    ap.add_argument("--out", default="data/anchor")
    a = ap.parse_args()
    src, out = Path(a.shapeshift_step2), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for dose in ("high", "low"):
        d = pd.read_parquet(src / f"morph_wells_{dose}.parquet", columns=COLS)
        d["is_dmso"] = d.is_dmso.astype(bool)
        d.loc[d.is_dmso, "compound_id"] = "DMSO"
        d.to_parquet(out / f"anchor_wells_{dose}.parquet", index=False)
        print(dose, d.shape, d.source.nunique(), "sources", d.plate.nunique(), "plates")
    u = pd.read_csv(src / "units.csv")
    u = u[u.tier == "T3_JUMP_U2OS"]
    rows = [(j, m) for ids, m in zip(u.dataset_ids, u.moa) for j in str(ids).split(";")]
    moa = pd.DataFrame(rows, columns=["compound_id", "moa"])
    moa = moa[moa.moa.notna() & (moa.moa != "unclear")]
    moa.to_csv(out / "moa.csv", index=False)
    print("moa", len(moa), "compounds,", moa.moa.nunique(), "classes")


if __name__ == "__main__":
    main()

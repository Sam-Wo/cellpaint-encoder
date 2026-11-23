"""Leakage rules (DESIGN.md 1.2, 4.1, R8): no anchor or eval-DMSO well and no held-out plate or
source in training; source_6 absent from the training manifest and from the dev cache."""

import numpy as np
import pandas as pd
import pytest

from cellpaint_encoder import CHANNELS
from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.manifest import build_manifest, check_manifest, split_plates


def fake_load_data(source, batch, plate, n_rows=4, n_cols=6, sites=3):
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            for s in range(1, sites + 1):
                w = f"{chr(65 + r)}{c + 1:02d}"
                d = {
                    "source": source,
                    "batch": batch,
                    "plate": plate,
                    "well": w,
                    "site": s,
                    "row": r,
                    "col": c,
                }
                d.update({f"key_{ch}": f"k/{plate}/{w}/{s}/{ch}" for ch in CHANNELS})
                rows.append(d)
    return pd.DataFrame(rows)


@pytest.fixture
def world():
    anchor, ld = [], {}
    for src in ("source_2", "source_3", "source_6"):
        for b in range(2):
            for p in range(3):
                plate = f"{src}_b{b}_p{p}"
                ld[plate] = fake_load_data(src, f"B{b}", plate)
                for i, w in enumerate(["A01", "A02", "B03", "C04"]):
                    anchor.append(
                        {
                            "source": src,
                            "batch": f"B{b}",
                            "plate": plate,
                            "well": w,
                            "compound_id": f"CPD{i}",
                            "is_dmso": False,
                        }
                    )
                for w in ["D01", "D02", "D03", "D04", "D05", "D06"]:
                    anchor.append(
                        {
                            "source": src,
                            "batch": f"B{b}",
                            "plate": plate,
                            "well": w,
                            "compound_id": "DMSO",
                            "is_dmso": True,
                        }
                    )
    return pd.DataFrame(anchor), ld


def cfg():
    return load_config(
        None,
        [
            "data.train_sources=[source_2, source_3]",
            "data.eval_sources=[source_2, source_3, source_6]",
            "data.n_eval_dmso_per_plate=4",
            "data.n_dev_dmso_per_plate=2",
            "data.norm_sites_per_plate=5",
            "data.sites_per_well=2",
            "data.heldout_plate_frac=0.34",
        ],
    ).data


def test_manifest_leakage_rules(world):
    anchor, ld = world
    m = build_manifest(cfg(), anchor, ld)
    tr = m[m.train]
    assert len(tr) and "source_6" not in set(tr.source)
    assert not tr.is_anchor.any() and not tr.eval_dmso.any()
    assert set(tr.plate_split) == {"train"}
    assert not (m.dev & (m.source == "source_6")).any()
    # 2 training sites per unmasked well
    assert (tr.groupby(["plate", "well"]).size() == 2).all()
    # eval DMSO: 4 per plate, dev DMSO the first 2 of them; norm sites among eval DMSO sites
    per_plate = m[m.eval_dmso].groupby("plate").well.nunique()
    assert (per_plate == 4).all()
    assert (m[m.dev_dmso].groupby("plate").well.nunique() == 2).all() and (m.dev_dmso <= m.eval_dmso).all()
    assert (m.norm <= m.eval_dmso).all() and (m[m.norm].groupby("plate").size() == 5).all()
    # all anchor wells of every eval source are in E1
    e1 = m[m.eval_set == "E1"]
    assert set(e1[e1.is_anchor].source) == {"source_2", "source_3", "source_6"}
    # source_6 plates are eval-only
    assert set(m[m.source == "source_6"].plate_split) == {"eval_only"}


def test_heldout_plates_stratified_by_batch(world):
    anchor, _ = world
    plates = anchor[anchor.source == "source_2"][["source", "batch", "plate"]].drop_duplicates()
    sp = split_plates(plates, 0.34, seed=0)
    held = [p for p, v in sp.items() if v == "heldout"]
    assert len(held) == 2 and len({p.split("_")[2] for p in held}) == 2  # one from each batch
    assert split_plates(plates, 0.34, seed=0) == sp  # deterministic


def test_manifest_is_seed_deterministic(world):
    anchor, ld = world
    a = build_manifest(cfg(), anchor, ld)
    b = build_manifest(cfg(), anchor, ld)
    pd.testing.assert_frame_equal(a, b)


def test_check_manifest_catches_leaks(world):
    anchor, ld = world
    m = build_manifest(cfg(), anchor, ld)
    bad = m.copy()
    bad.loc[bad.is_anchor.idxmax(), "train"] = True
    with pytest.raises(AssertionError):
        check_manifest(bad, cfg())
    bad = m.copy()
    bad.loc[np.flatnonzero(bad.source == "source_6")[0], "train"] = True
    with pytest.raises(AssertionError):
        check_manifest(bad, cfg())


def test_heldout_source_cannot_be_training_source(world):
    anchor, ld = world
    c = load_config(None, ["data.train_sources=[source_2, source_6]"]).data
    with pytest.raises(ValueError):
        build_manifest(c, anchor, ld)


def test_train_shard_index_has_no_masked_well(world, tmp_path):
    """What write_shards writes is a subset of the manifest's train rows (checked on the index)."""
    anchor, ld = world
    m = build_manifest(cfg(), anchor, ld)
    tr = m[m.train]
    idx = tr[["source", "plate", "well", "site"]].assign(tile=0)
    masked = m[m.is_anchor | m.eval_dmso | (m.source == "source_6")][["plate", "well"]]
    assert idx.merge(masked, on=["plate", "well"]).empty


def test_e2_e3_e4_rows_and_references(world):
    anchor, ld = world
    low = []
    for p in range(2):
        plate = f"source_7_p{p}"
        ld[plate] = fake_load_data("source_7", "R1", plate)
        for i, w in enumerate(["A01", "A02"]):
            low.append(
                {
                    "source": "source_7",
                    "batch": "R1",
                    "plate": plate,
                    "well": w,
                    "compound_id": f"CPD{i}",
                    "is_dmso": False,
                }
            )
        for w in ["D01", "D02", "D03", "D04", "D05"]:
            low.append(
                {
                    "source": "source_7",
                    "batch": "R1",
                    "plate": plate,
                    "well": w,
                    "compound_id": "DMSO",
                    "is_dmso": True,
                }
            )
    extra = pd.DataFrame(
        [
            {"source": "source_3", "batch": "T", "plate": "t2", "eval_set": "E3"},
            {"source": "source_3", "batch": "D", "plate": "dm", "eval_set": "E4"},
        ]
    )
    t2 = fake_load_data("source_3", "T", "t2")
    t2["compound_id"] = np.where(t2.col < 2, "JCP2022_033924", "T2CPD" + t2.col.astype(str))
    t2["is_dmso"] = t2.compound_id == "JCP2022_033924"
    ld["t2"], ld["dm"] = t2, fake_load_data("source_3", "D", "dm")
    m = build_manifest(cfg(), anchor, ld, anchor_low=pd.DataFrame(low), extra_plates=extra)
    for es in ("E2", "E3", "E4"):
        rows = m[m.eval_set == es]
        assert len(rows) and not rows.train.any() and not rows.dev.any()
        assert (rows.groupby("plate").eval_dmso.any()).all()  # every plate has a z-score reference
        assert m[(m.eval_set == es) & m.norm].plate.nunique() == rows.plate.nunique()
    assert set(m[m.eval_set == "E2"].source) == {"source_7"}
    assert m[m.eval_set == "E4"].is_dmso.all()
    assert (m[m.eval_set == "E3"].groupby("well").site.nunique() <= cfg().extra_sites_per_well).all()


def test_written_index_leak_check(world):
    from cellpaint_encoder.data.manifest import check_train_index

    anchor, ld = world
    c = cfg()
    m = build_manifest(c, anchor, ld)
    idx = m[m.train][["source", "plate", "well", "site"]]
    check_train_index(idx, m, c)  # clean
    leak = pd.concat([idx, m[m.is_anchor][["source", "plate", "well", "site"]].head(1)])
    with pytest.raises(AssertionError):
        check_train_index(leak, m, c)
    with pytest.raises(AssertionError):
        check_train_index(m[m.source == "source_6"][["source", "plate", "well", "site"]].head(2), m, c)

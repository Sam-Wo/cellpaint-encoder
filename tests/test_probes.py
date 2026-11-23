from cellpaint_encoder.eval.probes import position_probe, same_well_gap, source_probe
from synthetic_wells import make_wells


def test_planted_position_signal_detected():
    meta, X = make_wells(
        seed=0,
        cpd_signal=0.0,
        position_signal=3.0,
        n_dmso=40,
        shared_platemap=True,
        plates_per_source=5,
        sources=("source_2",),
    )
    # DMSO wells at random positions: plant a smooth row/col code so the probe can generalise
    import numpy as np

    rng = np.random.default_rng(0)
    code_r, code_c = rng.normal(size=(16, 32)) * 3, rng.normal(size=(24, 32)) * 3
    X = X + code_r[meta.row.to_numpy()] + code_c[meta.col.to_numpy()]
    dm = meta.is_dmso.to_numpy()
    r = position_probe(meta, X, dm)
    assert r["bal_acc_row"] > 3 * r["chance_row"] and r["bal_acc_col"] > 3 * r["chance_col"]
    assert r["edge_auc"] > 0.6


def test_no_position_signal_is_near_chance():
    meta, X = make_wells(
        seed=1, cpd_signal=0.0, position_signal=0.0, n_dmso=40, plates_per_source=5, sources=("source_2",)
    )
    r = position_probe(meta, X, meta.is_dmso.to_numpy())
    assert r["bal_acc_row"] < 2.5 * r["chance_row"] and r["bal_acc_col"] < 2.5 * r["chance_col"]
    assert 0.3 < r["edge_auc"] < 0.7


def test_too_few_plates_returns_nan_with_note():
    meta, X = make_wells(seed=2, plates_per_source=1, sources=("source_2",))
    r = position_probe(meta, X, meta.is_dmso.to_numpy())
    assert r["bal_acc_row"] != r["bal_acc_row"] and "note_row" in r  # NaN


def test_source_probe_and_same_well_gap():
    meta, X = make_wells(seed=3, source_signal=4.0, position_signal=2.0, n_dmso=20, shared_platemap=False)
    s = source_probe(meta, X)
    assert s["bal_acc_source"] > 0.9 and s["silhouette_source"] > 0
    g = same_well_gap(meta, X)
    assert set(g) >= {"within_same_well", "within_diff_well", "across_same_well", "across_diff_well"}

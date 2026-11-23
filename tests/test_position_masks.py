"""Pair masks of the pre-registration: M1 excludes same (source, well) and same plate; the strict
primary excludes the query's well ID; position signal inflates M2 but not M1 or the primary."""

import numpy as np

from cellpaint_encoder.eval.retrieval import cross_source, within_source
from synthetic_wells import make_wells


def test_m1_candidate_pool_excludes_same_position_and_plate():
    meta, X = make_wells(seed=2)
    _, _, q = within_source(meta, X, ["source_2"], position_restricted=True, n_null=10)
    src, plate, well = meta.source.to_numpy(), meta.plate.to_numpy(), meta.well.to_numpy()
    anchor = meta.is_anchor.to_numpy()
    for r in q.itertuples():
        i = r.query
        pool = anchor & (src == src[i]) & (plate != plate[i]) & (well != well[i])
        assert r.n_total == pool.sum()


def test_strict_cross_source_drops_same_well_id():
    meta, X = make_wells(seed=3, shared_platemap=True)  # same compound at the same well ID everywhere
    _, s, q = cross_source(meta, X, ["source_3"], ["source_2"], strict=False, n_null=10)
    _, s_strict, q_strict = cross_source(meta, X, ["source_3"], ["source_2"], strict=True, n_null=10)
    assert (q_strict.n_total < q.n_total).all()
    assert s_strict["n_groups"] == 0  # every positive shares the query's well ID: none left


def test_position_signal_inflates_m2_not_m1_or_primary():
    """Shared platemap within a source + strong within-source position effect, no compound signal."""
    meta, X = make_wells(
        seed=4, cpd_signal=0.0, position_signal=4.0, shared_platemap=True, plates_per_source=4
    )
    # replicate wells of a compound share (source, well) -> M2 positives are found by position
    _, m2, _ = within_source(meta, X, ["source_2", "source_3"], position_restricted=False, n_null=50)
    _, m1, _ = within_source(meta, X, ["source_2", "source_3"], position_restricted=True, n_null=50)
    _, prim, _ = cross_source(meta, X, ["source_3"], ["source_2", "source_8"], n_null=50)
    assert m2["mAP"] > 0.9
    assert m1["n_groups"] == 0 or m1["mAP"] < 0.5  # no different-position replicates to exploit
    assert prim["mAP"] < 0.5  # position effect is within-source: cross-source primary ignores it


def test_dmso_negative_variant_uses_dmso_pool():
    meta, X = make_wells(seed=5)
    _, _, q = within_source(meta, X, ["source_2"], position_restricted=True, dmso_negatives=True, n_null=10)
    dmso = meta.is_dmso.to_numpy()
    src = meta.source.to_numpy()
    plate, well = meta.plate.to_numpy(), meta.well.to_numpy()
    for r in q.itertuples():
        i = r.query
        n_dmso = (dmso & (src == src[i]) & (plate != plate[i]) & (well != well[i])).sum()
        assert r.n_total - r.n_pos == n_dmso
    assert np.isfinite(q.ap).any()

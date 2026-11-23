import numpy as np
import pytest

from cellpaint_encoder.eval.retrieval import average_precision, bh, cross_source, label_retrieval, null_ap
from synthetic_wells import make_wells


def test_average_precision_by_hand():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert average_precision(s, np.array([True, False, True, False])) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(s, np.array([True, True, False, False])) == 1.0
    assert np.isnan(average_precision(s, np.zeros(4, bool)))


def test_null_ap_mean():
    rng = np.random.default_rng(0)
    # expected AP of a random ranking with 1 positive among n is H_n / n
    n = 10
    assert null_ap(1, n, 20000, rng).mean() == pytest.approx(
        sum(1 / k for k in range(1, n + 1)) / n, rel=0.03
    )


def test_bh_monotone():
    q = bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert q[0] <= q[2] <= q[1] <= q[3] <= 1 and q[0] == pytest.approx(0.04)


def test_planted_replicates_map_near_one():
    meta, X = make_wells(cpd_signal=6.0, noise=0.5)
    _, s, _ = cross_source(meta, X, ["source_3"], ["source_2", "source_8"], n_null=200)
    assert s["mAP"] > 0.95 and s["frac_q05"] > 0.9


def test_shuffled_labels_near_null():
    meta, X = make_wells(cpd_signal=6.0, noise=0.5, shuffle_labels=True, seed=1)
    _, s, q = cross_source(meta, X, ["source_3"], ["source_2", "source_8"], n_null=200)
    expected = np.mean(
        [null_ap(r.n_pos, r.n_total, 300, np.random.default_rng(0)).mean() for r in q.itertuples() if r.n_pos]
    )
    assert abs(s["mAP"] - expected) < 0.1 and s["frac_q05"] < 0.3


def test_label_retrieval_planted_classes():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 16)) * 4
    lab = np.repeat(["a", "b", "c", "d"], 5).astype(object)
    X = centres[np.repeat(np.arange(4), 5)] + rng.normal(size=(20, 16))
    _, s, _ = label_retrieval(lab, X, n_null=100)
    assert s["mAP"] > 0.9 and s["n_classes"] == 4
    lab[:2] = "unclear"
    _, s2, _ = label_retrieval(lab, X, min_members=4, n_null=100)
    assert s2["n_classes"] == 3  # class 'a' now has 3 members

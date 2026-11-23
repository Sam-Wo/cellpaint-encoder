import numpy as np
import pandas as pd
import pytest

from cellpaint_encoder.eval.aggregate import sites_to_wells
from cellpaint_encoder.eval.profiles import fit_whitening, postprocess, robust_z
from cellpaint_encoder.eval.stats import holm, noninferiority, paired_compound_test


def test_sites_to_wells_mean():
    sites = pd.DataFrame(
        {
            "source": ["s"] * 4,
            "batch": ["b"] * 4,
            "plate": ["p"] * 4,
            "well": ["A01", "A01", "A02", "A02"],
            "site": [1, 2, 1, 2],
            "compound_id": ["x", "x", "y", "y"],
            "e0": [1.0, 3.0, 5.0, 7.0],
            "e1": [0.0, 2.0, 0.0, 0.0],
        }
    )
    meta, W = sites_to_wells(sites)
    assert list(meta.well) == ["A01", "A02"] and list(meta.n_sites) == [2, 2]
    assert np.allclose(W, [[2.0, 1.0], [6.0, 0.0]])
    assert "site" not in meta


def test_robust_z_per_plate_and_floor():
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"plate": ["a"] * 20 + ["b"] * 20})
    X = np.concatenate([rng.normal(10, 2, (20, 3)), rng.normal(-5, 2, (20, 3))])
    X[:, 2] = 1.0  # constant feature -> MAD 0 -> floor
    ref = np.tile([True] * 16 + [False] * 4, 2)
    Z, info = robust_z(meta, X, ref)
    assert abs(np.median(Z[:16, 0])) < 1e-6 and abs(np.median(Z[20:36, 0])) < 1e-6
    assert np.isfinite(Z).all() and np.abs(Z).max() <= 50
    assert info["scale_floored_frac"] > 0


def test_whitening_identity_cov_on_reference():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
    mu, W, info = fit_whitening(D, var_keep=1.0, eps=0.0)
    V = (D - mu) @ W
    assert np.allclose(np.cov(V, rowvar=False), np.eye(info["sphere_dim"]), atol=1e-6)
    meta = pd.DataFrame({"plate": ["a"] * 250 + ["b"] * 250})
    out, pinfo = postprocess(meta, D, np.ones(500, bool), eps=1.0)
    assert out.shape[0] == 500 and pinfo["whiten"]["sphere_dim"] <= 6


def test_bootstrap_ci_and_wilcoxon():
    rng = np.random.default_rng(0)
    idx = [f"c{i}" for i in range(171)]
    a = pd.Series(rng.uniform(0.2, 0.6, 171), index=idx)
    b = a - 0.05 + rng.normal(0, 0.05, 171)  # ours better by 0.05
    r = paired_compound_test(a, b, n_boot=2000)
    assert r["n_compounds"] == 171 and r["ci_low"] > 0 and r["p_wilcoxon"] < 1e-6 and r["ci_excludes_0"]
    null = paired_compound_test(a, a + rng.normal(0, 0.05, 171), n_boot=2000)
    assert null["ci_low"] < 0 < null["ci_high"]
    ni = noninferiority(a, a + 0.005, margin=0.01, n_boot=2000)
    assert ni["noninferior"]


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert (
        adj["a"] == pytest.approx(0.03)
        and adj["c"] == pytest.approx(0.06)
        and adj["b"] == pytest.approx(0.06)
    )
    assert np.isnan(adj["d"])

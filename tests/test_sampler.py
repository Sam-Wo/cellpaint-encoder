import numpy as np
import pandas as pd
import pytest

from cellpaint_encoder.data.sampler import BalancedBatchSampler


def index(n_by_source, dmso_frac=0.3, n_plates=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, n in n_by_source.items():
        for _ in range(n):
            rows.append(
                {"source": s, "is_dmso": rng.random() < dmso_frac, "plate": f"{s}_{rng.integers(n_plates)}"}
            )
    return pd.DataFrame(rows)


def test_equal_per_source_and_dmso_cap():
    idx = index({"source_2": 4000, "source_3": 3000, "source_8": 2000, "source_10": 500}, dmso_frac=0.5)
    s = BalancedBatchSampler(
        idx,
        256,
        seed=0,
        micro_per_epoch=50,
        sources=["source_2", "source_3", "source_8", "source_10"],
        dmso_cap=0.2,
    )
    for m in (0, 7, 49, 50, 123):
        b = idx.iloc[s.batch(m)]
        assert len(b) == 256
        assert (b.source.value_counts() == 64).all()  # 64 from each source
        assert b.is_dmso.mean() <= 0.2 + 1e-9  # DMSO capped at 20 %
        assert b.plate.nunique() >= 16  # spans many plates
    # natural-ish DMSO fraction below the cap is kept
    idx2 = index({"a": 1000, "b": 1000}, dmso_frac=0.05)
    s2 = BalancedBatchSampler(idx2, 100, seed=0, micro_per_epoch=10, dmso_cap=0.2)
    assert 0 < idx2.iloc[s2.batch(0)].is_dmso.mean() <= 0.1


def test_small_source_is_oversampled_without_repeats_within_a_cycle():
    idx = index({"big": 2000, "small": 100}, dmso_frac=0.0)
    s = BalancedBatchSampler(idx, 20, seed=1, micro_per_epoch=100)
    seen = np.concatenate([s.batch(m) for m in range(10)])  # 100 small-source draws = one full cycle
    small = seen[idx.source.to_numpy()[seen] == "small"]
    assert len(small) == 100 and len(np.unique(small)) == 100


def test_stateless_resume_and_seed_dependence():
    idx = index({"a": 500, "b": 300})
    s = BalancedBatchSampler(idx, 32, seed=3, micro_per_epoch=20)
    full = [b for _, b in zip(range(30), iter(s))]
    resumed = list(BalancedBatchSampler(idx, 32, seed=3, micro_per_epoch=20, start=17, stop=30))
    assert full[17:] == resumed
    other = BalancedBatchSampler(idx, 32, seed=4, micro_per_epoch=20)
    assert other.batch(0) != s.batch(0)
    assert s.batch(0) != s.batch(20)  # new epoch, new permutation


def test_natural_mode_follows_proportions():
    idx = index({"a": 9000, "b": 1000}, dmso_frac=0.0)
    s = BalancedBatchSampler(idx, 1000, seed=0, micro_per_epoch=5, balance="natural")
    frac_a = (idx.iloc[s.batch(0)].source == "a").mean()
    assert 0.85 < frac_a < 0.95


def test_missing_source_raises():
    with pytest.raises(ValueError):
        BalancedBatchSampler(index({"a": 10}), 4, 0, 2, sources=["a", "zzz"])

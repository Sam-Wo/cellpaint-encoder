import numpy as np
import torch

from cellpaint_encoder.config import AugCfg
from cellpaint_encoder.data.augment_gpu import MultiCropAugment, gaussian_blur, intensity_aug, random_crops
from cellpaint_encoder.data.shards import ShardReader, TileDataset, iter_shard, load_index
from conftest import write_synthetic_shards


def test_shard_roundtrip_random_and_sequential(tmp_path):
    idx = write_synthetic_shards(tmp_path / "s", n_per_source=6)
    assert idx.shard.nunique() > 1  # rolled over max_bytes
    r = ShardReader(tmp_path / "s")
    seq = {}
    for shard in idx.shard.unique():
        for k, a, meta in iter_shard(tmp_path / "s" / shard):  # WebDataset-style sequential read
            seq[k] = a
            assert meta["source"] in ("source_2", "source_3")
    for row in idx.itertuples():
        assert np.array_equal(r.read(row.shard, row.offset, row.size), seq[row.key])
    ds = TileDataset(load_index(tmp_path / "s"), tmp_path / "s", cache_mb=1)
    a, b = ds[0], ds[0]
    assert a.dtype == torch.uint8 and torch.equal(a, b) and ds.reader.hits == 1
    import pickle

    ds2 = pickle.loads(pickle.dumps(ds))  # DataLoader workers (spawn) need a picklable dataset
    assert torch.equal(ds2[3], ds[3])


def test_crop_shapes_and_determinism():
    tiles = torch.randint(0, 255, (3, 5, 64, 64), dtype=torch.uint8)
    aug = MultiCropAugment(AugCfg(), 32, 16, 4, 0.2, [0.2] * 5, [0.2] * 5)
    c1 = aug(tiles, torch.Generator().manual_seed(5))
    c2 = aug(tiles, torch.Generator().manual_seed(5))
    assert [tuple(c.shape) for c in c1] == [(3, 5, 32, 32)] * 2 + [(3, 5, 16, 16)] * 4
    assert all(torch.equal(a, b) for a, b in zip(c1, c2))
    c3 = aug(tiles, torch.Generator().manual_seed(6))
    assert not torch.equal(c1[0], c3[0])


def test_d4_crop_is_exact_permutation_of_a_window():
    """No scale jitter, full-tile crop: output is the tile up to a flip/rotation (no resampling)."""
    x = torch.arange(2 * 16 * 16, dtype=torch.float32).view(1, 2, 16, 16)
    out = random_crops(x, 16, 0.0, torch.Generator().manual_seed(0))
    cands = [torch.rot90(t, k, (2, 3)) for t in (x, torch.flip(x, (3,))) for k in range(4)]
    assert any(torch.allclose(out, c, atol=1e-4) for c in cands)


def test_intensity_aug_channel_dropout_and_no_mixing():
    cfg = AugCfg(p_intensity=0, p_noise=0, p_channel_dropout=1.0, p_blur_global1=0)
    x = torch.rand(4, 5, 8, 8) * 0.5 + 0.25
    y = intensity_aug(x, cfg, 0.0, torch.Generator().manual_seed(0))
    zeroed = (y.flatten(2).abs().sum(-1) == 0).sum(1)
    assert (zeroed == 1).all()  # exactly one channel dropped per crop
    # channels are never mixed: a single bright channel stays in its channel
    z = torch.zeros(1, 5, 8, 8)
    z[:, 2] = 1.0
    out = intensity_aug(z, AugCfg(p_channel_dropout=0, p_noise=0), 1.0, torch.Generator().manual_seed(1))
    assert out[:, [0, 1, 3, 4]].abs().max() < 0.2  # only the offset can touch other channels


def test_blur_identity_when_sigma_zero():
    x = torch.rand(2, 3, 16, 16)
    y = gaussian_blur(x, torch.tensor([0.0, 1.0]))
    assert torch.allclose(y[0], x[0]) and not torch.allclose(y[1], x[1])
    assert abs(float(y[1].mean() - x[1].mean())) < 0.05

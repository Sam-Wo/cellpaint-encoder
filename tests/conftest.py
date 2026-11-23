"""Shared fixtures: tiny configs and synthetic shards (no downloads; CPU only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.shards import ShardWriter

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]

torch.set_num_threads(2)


def tiny_overrides(shards: Path, run_dir: Path | None = None) -> list[str]:
    ov = [
        f"data.shards_dir={shards.as_posix()}",
        "data.train_sources=[source_2, source_3]",
        "data.num_workers=0",
        "data.pin_memory=false",
        "data.channel_mean=[0.2,0.2,0.2,0.2,0.2]",
        "data.channel_std=[0.2,0.2,0.2,0.2,0.2]",
        "model.embed_dim=64",
        "model.depth=2",
        "model.num_heads=2",
        "model.init=random",
        "model.drop_path=0.1",
        "dino.out_dim=128",
        "dino.hidden_dim=64",
        "dino.bottleneck_dim=32",
        "dino.global_crop_size=32",
        "dino.local_crop_size=16",
        "dino.n_local_crops=2",
        "dino.warmup_teacher_temp_epochs=1",
        "optim.epochs=2",
        "optim.batch_size=4",
        "optim.accum_steps=2",
        "optim.warmup_epochs=0.5",
        "run.device=cpu",
        "run.precision=fp32",
        "run.epoch_size=8",
        "run.log_every_steps=1",
        "run.ckpt_every_minutes=0",
        "log.tensorboard=false",
    ]
    if run_dir is not None:
        ov.append(f"run.out_dir={run_dir.as_posix()}")
    return ov


def write_synthetic_shards(
    path: Path,
    n_per_source: int = 8,
    size: int = 64,
    seed: int = 0,
    sources=("source_2", "source_3"),
    dmso_every: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = ShardWriter(path, "train", max_bytes=40_000, zstd_level=1)
    for s in sources:
        for i in range(n_per_source):
            arr = rng.integers(0, 255, (5, size, size), dtype=np.uint8)
            meta = {
                "source": s,
                "batch": "b",
                "plate": f"{s}_p{i % 3}",
                "well": f"A{i + 1:02d}",
                "site": 1,
                "row": 0,
                "col": i,
                "compound_id": "",
                "is_dmso": i % dmso_every == 0,
                "tile": 0,
            }
            w.write(f"{s}__{meta['plate']}__{meta['well']}__s1__t0", arr, meta)
    return w.close()


@pytest.fixture
def synthetic_shards(tmp_path):
    p = tmp_path / "shards"
    write_synthetic_shards(p)
    return p


@pytest.fixture
def tiny_cfg(synthetic_shards, tmp_path):
    return load_config(None, tiny_overrides(synthetic_shards, tmp_path / "runs"))

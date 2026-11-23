import numpy as np
import pandas as pd
import pytest
import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.train.loop import Trainer, clip_gradients
from conftest import tiny_overrides


def test_three_steps_accum2(tiny_cfg, tmp_path):
    tr = Trainer(tiny_cfg, tmp_path / "r", install_signals=False)
    before = {k: v.clone() for k, v in tr.student.state_dict().items()}
    out = tr.train(resume=None)
    assert out["stopped"] == "schedule_done" and out["step"] == tr.total_steps == 4
    log = pd.read_csv(tmp_path / "r" / "train_log.csv")
    assert len(log) == 4 and np.isfinite(log.loss).all()
    assert (log.effective_batch == 4).all()
    changed = any(not torch.equal(before[k], v) for k, v in tr.student.state_dict().items())
    assert changed
    assert (tmp_path / "r" / "teacher_backbone.pt").exists()


def test_loss_decreases_on_fixed_data(synthetic_shards, tmp_path):
    """Teacher and center frozen (momentum 1), constant LR, no augmentation noise: the student learns to match
    the teacher's sharpened targets, so the DINO loss goes down. (With an EMA teacher the loss is
    not monotone: centering raises the teacher entropy, which is part of the loss.)"""
    ov = tiny_overrides(synthetic_shards, tmp_path) + [
        "optim.epochs=15",
        "optim.warmup_epochs=0",
        "optim.base_lr=0.005",
        "optim.min_lr=0.005",
        "optim.weight_decay=0.0",
        "optim.weight_decay_end=0.0",
        "dino.freeze_last_layer_epochs=0",
        "dino.warmup_teacher_temp_epochs=0",
        "dino.aug.p_intensity=0",
        "dino.aug.p_noise=0",
        "dino.aug.p_channel_dropout=0",
        "dino.aug.p_blur_global1=0",
        "dino.aug.p_blur_local=0",
        "dino.momentum_teacher=1.0",
        "dino.center_momentum=1.0",
        "model.drop_path=0.0",
        "run.epoch_size=4",
    ]
    cfg = load_config(None, ov)
    out = Trainer(cfg, tmp_path / "dec", install_signals=False).train(resume=None)
    log = pd.read_csv(tmp_path / "dec" / "train_log.csv")
    assert out["step"] == 15
    assert log.loss.iloc[-3:].mean() < log.loss.iloc[:3].mean()


def test_bf16_autocast_path_on_cpu(tiny_cfg, tmp_path):
    cfg = load_config(
        None,
        [
            *tiny_overrides(__import__("pathlib").Path(tiny_cfg.data.shards_dir), tmp_path),
            "run.precision=bf16",
            "run.max_steps=1",
        ],
    )
    out = Trainer(cfg, tmp_path / "bf", install_signals=False).train(resume=None)
    assert out["step"] == 1


def test_per_param_clipping():
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(3))
    a.grad = torch.tensor([30.0, 40.0, 0.0])  # norm 50
    b.grad = torch.tensor([0.3, 0.4, 0.0])  # norm 0.5
    total = clip_gradients([a, b], 3.0, "per_param")
    assert total == pytest.approx((50**2 + 0.5**2) ** 0.5, rel=1e-5)
    assert float(a.grad.norm()) == pytest.approx(3.0, rel=1e-4)
    assert float(b.grad.norm()) == pytest.approx(0.5, rel=1e-6)  # untouched


def test_in_training_dev_eval_hook(synthetic_shards, tmp_path):
    """The trainer's dev hook embeds the dev cache with the EMA teacher and logs dev/ metrics."""
    from cellpaint_encoder.data.shards import ShardWriter
    from cellpaint_encoder.eval.pipeline import dev_eval_hook

    rng = np.random.default_rng(0)
    w = ShardWriter(tmp_path / "dev", "dev", 1e9, 1)
    for s in ("source_2", "source_3", "source_8"):
        for p in range(2):
            for i, cpd in enumerate(["C0", "C1", "C2", "DMSO", "DMSO"]):
                meta = {
                    "source": s,
                    "batch": "b",
                    "plate": f"{s}_{p}",
                    "well": f"A{i + 1:02d}",
                    "site": 1,
                    "row": 0,
                    "col": i,
                    "compound_id": cpd,
                    "is_dmso": cpd == "DMSO",
                    "is_anchor": cpd != "DMSO",
                    "eval_dmso": cpd == "DMSO",
                    "plate_split": "train",
                }
                w.write(f"{s}_{p}_{i}", rng.integers(0, 255, (5, 64, 64), dtype=np.uint8), meta)
    w.close()
    ov = tiny_overrides(synthetic_shards, tmp_path) + [
        f"data.dev_shards_dir={(tmp_path / 'dev').as_posix()}",
        "eval.dev_every_epochs=1",
        "eval.grid=2",
        "eval.crop=32",
        "eval.n_null=20",
        "optim.epochs=1",
    ]
    cfg = load_config(None, ov)
    Trainer(cfg, tmp_path / "dv", install_signals=False).train(resume=None, dev_eval=dev_eval_hook)
    dev = pd.read_csv(tmp_path / "dv" / "dev_log.csv")
    assert (
        len(dev) == 1
        and "dev/cross_source_mAP" in dev.columns
        and np.isfinite(dev["dev/cross_source_mAP"]).all()
    )
    train = pd.read_csv(tmp_path / "dv" / "train_log.csv")
    assert train.loss.notna().all()

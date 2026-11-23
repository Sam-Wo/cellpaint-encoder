"""Train 4 steps straight vs 2 + resume + 2: bitwise-identical state on CPU (DESIGN.md 3.3), with
accum=2 and drop-path > 0 so RNG and accumulation are both exercised. Also: auto-resume skips a
truncated checkpoint, and the CSV log has exactly one row per step after resuming."""

import pandas as pd
import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.train.checkpoint import find_resume
from cellpaint_encoder.train.loop import Trainer, state_equal
from conftest import tiny_overrides


def _cfg(shards, tmp, extra=()):
    return load_config(
        None, [*tiny_overrides(shards, tmp), "run.ckpt_every_steps=1", "run.keep_last=10", *extra]
    )


def test_resume_bitwise(synthetic_shards, tmp_path):
    cfg = _cfg(synthetic_shards, tmp_path)
    a = Trainer(cfg, tmp_path / "a", install_signals=False)
    a.train(resume=None)
    st_a = torch.load(tmp_path / "a" / "ckpt-step000000004.pt", weights_only=True)

    cfg_stop = _cfg(synthetic_shards, tmp_path, ["run.max_steps=2"])  # non-science key: same hash
    b1 = Trainer(cfg_stop, tmp_path / "b", install_signals=False)
    out = b1.train(resume=None)
    assert out["stopped"] == "max_steps" and out["step"] == 2
    del b1
    b2 = Trainer(cfg, tmp_path / "b", install_signals=False)
    b2.train(resume="auto")
    assert b2.resumed_from.endswith("ckpt-step000000002.pt")
    st_b = torch.load(tmp_path / "b" / "ckpt-step000000004.pt", weights_only=True)
    eq = state_equal(st_a, st_b)
    assert all(eq.values()), eq
    la, lb = pd.read_csv(tmp_path / "a" / "train_log.csv"), pd.read_csv(tmp_path / "b" / "train_log.csv")
    assert list(lb.step) == [1, 2, 3, 4]
    assert (la.loss.values == lb.loss.values).all() and (la.grad_norm.values == lb.grad_norm.values).all()


def test_resume_skips_corrupt_checkpoint(synthetic_shards, tmp_path):
    cfg = _cfg(synthetic_shards, tmp_path, ["run.max_steps=3"])
    Trainer(cfg, tmp_path / "c", install_signals=False).train(resume=None)
    newest = tmp_path / "c" / "ckpt-step000000003.pt"
    newest.write_bytes(newest.read_bytes()[:1000])  # simulate a kill mid-write without the rename
    p, st = find_resume(tmp_path / "c")
    assert p.name == "ckpt-step000000002.pt" and st["step"] == 2


def test_resume_refuses_other_config(synthetic_shards, tmp_path):
    cfg = _cfg(synthetic_shards, tmp_path, ["run.max_steps=1"])
    Trainer(cfg, tmp_path / "d", install_signals=False).train(resume=None)
    other = _cfg(synthetic_shards, tmp_path, ["optim.base_lr=0.1"])
    t = Trainer(other, tmp_path / "d", install_signals=False)
    try:
        t.maybe_resume("auto")
        raise AssertionError("should refuse a checkpoint from another config")
    except ValueError:
        pass


def test_sigterm_stop_checkpoints_and_resumes_identically(synthetic_shards, tmp_path):
    """The pre-emption path pod.sh relies on: stop flag -> checkpoint at that step -> 'signal'
    (train.py turns it into exit code 75) -> resume -> same final state as an uninterrupted run."""
    cfg = _cfg(synthetic_shards, tmp_path, ["run.ckpt_every_steps=0", "run.keep_last=2"])
    ref = Trainer(cfg, tmp_path / "ref", install_signals=False)
    ref.train(resume=None)
    st_ref = ref.state_dict()

    t1 = Trainer(cfg, tmp_path / "pre", install_signals=False)
    t1.stop.flag = True  # as if SIGTERM arrived during the first step
    out = t1.train(resume=None)
    assert out["stopped"] == "signal" and out["step"] == 1
    assert (tmp_path / "pre" / "ckpt-step000000001.pt").exists()
    t2 = Trainer(cfg, tmp_path / "pre", install_signals=False)
    assert t2.train(resume="auto")["stopped"] == "schedule_done"
    eq = state_equal(st_ref, t2.state_dict())
    assert all(eq.values()), eq

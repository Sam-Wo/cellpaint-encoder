"""Per-iteration schedules (DINO style), indexed by the global optimiser step so resume is exact.

>>> s = cosine_schedule(1.0, 0.0, epochs=2, iters_per_epoch=5, warmup_epochs=1)
>>> len(s), float(s[0]), float(s[5]), float(s[-1]) < 0.1
(10, 0.0, 1.0, True)
>>> t = teacher_temp_schedule(0.04, 0.07, warmup_epochs=1, epochs=2, iters_per_epoch=4)
>>> [round(float(x), 3) for x in t]
[0.04, 0.048, 0.055, 0.062, 0.07, 0.07, 0.07, 0.07]
"""

from __future__ import annotations

import numpy as np


def cosine_schedule(
    base: float,
    final: float,
    epochs: int,
    iters_per_epoch: int,
    warmup_epochs: float = 0,
    start_warmup: float = 0.0,
) -> np.ndarray:
    total = int(epochs * iters_per_epoch)
    n_warm = min(int(round(warmup_epochs * iters_per_epoch)), total)
    warm = np.linspace(start_warmup, base, n_warm, endpoint=False) if n_warm else np.array([])
    iters = np.arange(total - n_warm)
    denom = max(len(iters), 1)
    cos = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / denom))
    out = np.concatenate([warm, cos])
    assert len(out) == total
    return out


def teacher_temp_schedule(
    warmup_temp: float, temp: float, warmup_epochs: float, epochs: int, iters_per_epoch: int
) -> np.ndarray:
    """Linear warm-up of the teacher temperature (0.04 -> 0.07 over 30 epochs), then constant."""
    total = int(epochs * iters_per_epoch)
    n_warm = min(int(round(warmup_epochs * iters_per_epoch)), total)
    warm = np.linspace(warmup_temp, temp, n_warm, endpoint=False) if n_warm else np.array([])
    return np.concatenate([warm, np.full(total - n_warm, temp)])


def build_schedules(cfg, iters_per_epoch: int) -> dict[str, np.ndarray]:
    o, d = cfg.optim, cfg.dino
    lr = o.base_lr * o.batch_size / 256.0
    return {
        "lr": cosine_schedule(lr, o.min_lr, o.epochs, iters_per_epoch, o.warmup_epochs),
        "wd": cosine_schedule(o.weight_decay, o.weight_decay_end, o.epochs, iters_per_epoch),
        "momentum": cosine_schedule(d.momentum_teacher, 1.0, o.epochs, iters_per_epoch),
        "teacher_temp": teacher_temp_schedule(
            d.warmup_teacher_temp, d.teacher_temp, d.warmup_teacher_temp_epochs, o.epochs, iters_per_epoch
        ),
    }

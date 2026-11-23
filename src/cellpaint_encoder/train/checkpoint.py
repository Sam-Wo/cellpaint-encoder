"""Atomic checkpoints, retention, ``--resume auto`` and a SIGTERM handler (DESIGN.md 3.3).

Checkpoint contents: student, teacher (backbone + head each), optimiser, grad scaler, DINO
center, global step, epoch, sampler position (micro-batch index; the sampler is stateless
otherwise), RNG states (Python, numpy, torch, CUDA), the resolved config and the git SHA.

Written as ``<name>.tmp`` then renamed. Rolling checkpoints ``ckpt-step<N>.pt`` keep the last
``keep_last``; epoch checkpoints ``ckpt-epoch<E>.pt`` every ``keep_every_epochs`` are permanent.
Loading uses ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path

import torch

ROLLING = re.compile(r"ckpt-step(\d+)\.pt$")
EPOCH = re.compile(r"ckpt-epoch(\d+)\.pt$")


def git_sha(root: Path | None = None) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return (sha or "unknown") + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    state: dict, run_dir: Path, step: int, keep_last: int, epoch_tag: int | None = None
) -> Path:
    run_dir = Path(run_dir)
    p = run_dir / f"ckpt-step{step:09d}.pt"
    atomic_save(state, p)
    if epoch_tag is not None:
        atomic_save(state, run_dir / f"ckpt-epoch{epoch_tag:04d}.pt")
    rolling = sorted((int(ROLLING.search(f.name).group(1)), f) for f in run_dir.glob("ckpt-step*.pt"))
    for _, f in rolling[:-keep_last] if keep_last > 0 else []:
        f.unlink(missing_ok=True)
    return p


def list_checkpoints(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    found = []
    for f in run_dir.glob("ckpt-*.pt"):
        m = ROLLING.search(f.name)
        if m:
            found.append((int(m.group(1)), 1, f))
    return [f for _, _, f in sorted(found, reverse=True)]


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def find_resume(run_dir: Path) -> tuple[Path | None, dict | None]:
    """Newest checkpoint that loads and is complete (a truncated file is skipped)."""
    for p in list_checkpoints(run_dir):
        try:
            st = load_checkpoint(p)
        except Exception:  # noqa: BLE001 - corrupt/partial file: try the next one
            continue
        if st.get("complete"):
            return p, st
    return None, None


class StopFlag:
    """Set by SIGTERM/SIGINT (spot or community-pod pre-emption); the loop checkpoints and exits."""

    def __init__(self, install: bool = True):
        self.flag = False
        self.signal = None
        if install:
            for s in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(s, self._handler)
                except (ValueError, OSError):  # not in the main thread
                    pass

    def _handler(self, signum, _frame):
        self.flag, self.signal = True, signum

    def __bool__(self):
        return self.flag

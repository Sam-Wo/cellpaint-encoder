"""CSV + TensorBoard (+ optional W&B) logging.

One CSV per stream: ``train_log.csv`` (every ``log_every_steps``) and ``dev_log.csv`` (metrics
whose names start with ``dev/``). On resume every CSV is truncated to the resumed step, so a
killed-and-resumed run has exactly one row per logged step."""

from __future__ import annotations

import csv
from pathlib import Path


class RunLogger:
    STREAMS = ("train", "dev")

    def __init__(self, run_dir: Path, cfg_log, resume_step: int | None, config_dict: dict | None = None):
        self.run_dir = Path(run_dir)
        self.enabled_csv = cfg_log.csv
        self.fields: dict[str, list[str]] = {}
        self.tb = None
        self.wandb = None
        if self.enabled_csv:
            for s in self.STREAMS:
                self._truncate(s, resume_step)
        if cfg_log.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.run_dir / "tb"), purge_step=resume_step)
            except ImportError:
                print("tensorboard not installed: TensorBoard logging off")
        if cfg_log.wandb:
            import wandb  # optional dependency

            self.wandb = wandb.init(
                project=cfg_log.wandb_project,
                dir=str(self.run_dir),
                config=config_dict,
                name=self.run_dir.name,
                resume="allow",
                id=self.run_dir.name,
            )

    def path(self, stream: str) -> Path:
        return self.run_dir / f"{stream}_log.csv"

    def _truncate(self, stream: str, resume_step):
        p = self.path(stream)
        if not p.exists():
            return
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        self.fields[stream] = list(rows[0].keys())
        keep = [r for r in rows if resume_step is not None and int(r["step"]) <= resume_step]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields[stream])
            w.writeheader()
            w.writerows(keep)

    def log(self, step: int, metrics: dict) -> None:
        stream = "dev" if metrics and all(k.startswith("dev/") for k in metrics) else "train"
        row = {"step": step, **metrics}
        if self.enabled_csv:
            p = self.path(stream)
            new = not p.exists()
            fields = self.fields.setdefault(stream, list(row.keys()))
            with open(p, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()

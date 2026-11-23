"""CSV + TensorBoard (+ optional W&B) logging. On resume the CSV is truncated to the resumed step,
so a killed-and-resumed run has exactly one row per step."""

from __future__ import annotations

import csv
from pathlib import Path


class RunLogger:
    def __init__(
        self,
        run_dir: Path,
        cfg_log,
        resume_step: int | None,
        config_dict: dict | None = None,
        name: str = "train",
    ):
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / f"{name}_log.csv"
        self.fields: list[str] | None = None
        self.tb = None
        self.wandb = None
        if cfg_log.csv:
            self._truncate(resume_step)
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
        self.enabled_csv = cfg_log.csv

    def _truncate(self, resume_step):
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        self.fields = list(rows[0].keys())
        keep = [r for r in rows if resume_step is not None and int(r["step"]) <= resume_step]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(keep)

    def log(self, step: int, metrics: dict) -> None:
        row = {"step": step, **metrics}
        if self.enabled_csv:
            new = not self.csv_path.exists()
            if self.fields is None:
                self.fields = list(row.keys())
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
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

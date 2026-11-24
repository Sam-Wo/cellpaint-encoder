"""Train the DINO encoder (DESIGN.md 3).

    py -3.13 -I -X utf8 scripts/train.py --config configs/pilot.yaml [--resume auto] [key=value ...]

The run directory defaults to ``runs/<name>-<config hash>``; ``--resume auto`` (the default)
continues from its newest valid checkpoint. The trainer refuses shards whose ``stats.json``
records other pixel-size resampling settings (R7) than the config; so does the dev hook.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json

from cellpaint_encoder.config import load_config
from cellpaint_encoder.train.loop import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", help="auto | none | path to a checkpoint")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--no-dev-eval", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    tr = Trainer(cfg, a.run_dir)
    dev = None
    if (
        not a.no_dev_eval
        and cfg.eval.dev_every_epochs
        and Path(cfg.data.dev_shards_dir, "index.parquet").exists()
    ):
        from cellpaint_encoder.data.resample import check_shards_resample
        from cellpaint_encoder.eval.pipeline import dev_eval_hook

        # R7: fail now, not at the first dev evaluation
        check_shards_resample(cfg.data.dev_shards_dir, cfg.data, "dev shards")
        dev = dev_eval_hook
    out = tr.train(resume=None if a.resume == "none" else a.resume, dev_eval=dev)
    print(json.dumps(out, indent=1))
    if out["stopped"] == "signal":  # pre-empted: a wrapper loop should restart with --resume auto
        sys.exit(75)


if __name__ == "__main__":
    main()

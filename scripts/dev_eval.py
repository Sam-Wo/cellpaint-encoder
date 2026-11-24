"""Dev-cache evaluation of one encoder, exactly as the trainer's in-training hook does it.

    python scripts/dev_eval.py --config configs/full.yaml --encoder imagenet_dino
    python scripts/dev_eval.py --config configs/full.yaml --encoder ours=runs/full_seed0/ckpt-step000001395.pt

Used for gate G1(c): our dev cross-source mAP must be >= ImageNet-frozen + 0.01 on the same dev
cache, the same wells and the same post-processing. source_6 is never in the dev cache.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import json

import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.data.resample import check_shards_resample
from cellpaint_encoder.embed.run import embed_dev_shards
from cellpaint_encoder.eval.pipeline import evaluate_sites
from cellpaint_encoder.train.loop import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--encoder", required=True, help="imagenet_dino | random_init | ours=<ckpt or teacher_backbone.pt>"
    )
    ap.add_argument("--out", default=None, help="optional json output")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    if cfg.run.threads:
        torch.set_num_threads(cfg.run.threads)
    device = resolve_device(cfg.run.device)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from embed import build_encoder  # scripts/embed.py

    check_shards_resample(cfg.data.dev_shards_dir, cfg.data, "dev shards")  # R7
    enc = build_encoder(a.encoder, cfg, device, None)
    sites = embed_dev_shards(enc, cfg.data.dev_shards_dir, cfg.eval.batch_size)
    r = evaluate_sites(sites, cfg, which=("primary", "m1", "probes"))
    out = {
        "encoder": a.encoder,
        "dev_cross_source_mAP": r["dev_cross_source"]["mAP"],
        "by_query_source": r["dev_cross_source"]["by_query_source"],
        "M1_mAP": r["M1"]["mAP"],
        "M2_mAP": r["M2"]["mAP"],
        "eps": r["eps"],
        "position_probe": r["position_probe"],
        "n_wells": r["n_wells"],
    }
    print(json.dumps(out, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

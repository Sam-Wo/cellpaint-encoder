"""Evaluate gate G1 (pilot) or G2 (25 % of the main run) from a run's logs (DESIGN.md 5.4).

    python scripts/check_gates.py g1 --run runs/full_seed0 --imagenet-dev results/dev_imagenet.json \
        [--gpu-util runs/full_seed0/gpu_util.csv]
    python scripts/check_gates.py g2 --run runs/full_seed0 --cellprofiler-dev results/dev/results.json

Prints every criterion with its measured value and PASS/FAIL, and the plan tier implied by the
measured throughput (G1 rule: >=250/s full plan; 200-250 seed 1 at 50 epochs + 2 ablations;
150-200 one seed + A1; <150 debug/switch GPU or cut-down).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def tier(sps: float) -> str:
    if sps >= 250:
        return "full plan (2 seeds x 100 epochs + 3 ablations)"
    if sps >= 200:
        return "seed 1 cut to 50 epochs, 2 ablations"
    if sps >= 150:
        return "1 seed + A1 only"
    return "below 150/s: debug the loader or switch to H100, else run the cut-down plan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--imagenet-dev", help="json from scripts/dev_eval.py --encoder imagenet_dino")
    ap.add_argument("--cellprofiler-dev", help="results.json from scripts/evaluate.py on the dev set")
    ap.add_argument(
        "--gpu-util", help="nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 10"
    )
    ap.add_argument("--skip-steps", type=int, default=200, help="warm-up steps excluded from throughput")
    a = ap.parse_args()
    run = Path(a.run)
    log = pd.read_csv(run / "train_log.csv")
    dev = pd.read_csv(run / "dev_log.csv") if (run / "dev_log.csv").exists() else pd.DataFrame()
    steady = log[log.step > a.skip_steps] if (log.step > a.skip_steps).any() else log
    res = {}
    sps = float(steady.samples_per_s.median())
    res["throughput_samples_per_s"] = (sps, sps >= 250)
    res["data_wait_frac"] = (
        float(steady.data_wait_frac.median()),
        float(steady.data_wait_frac.median()) < 0.10,
    )
    if a.gpu_util and Path(a.gpu_util).exists():
        u = pd.read_csv(a.gpu_util, header=None).iloc[:, 0].astype(float)
        res["gpu_busy_pct_median"] = (float(u.median()), float(u.median()) >= 85)
    last = log.iloc[-1]
    ent, emax = float(last.teacher_entropy), float(last.max_entropy)
    res["teacher_entropy_not_at_extremes"] = (ent, 0.05 * emax < ent < 0.98 * emax)
    f0, f1 = float(log.feat_std.iloc[0]), float(last.feat_std)
    res["feat_std_vs_initial"] = (f1 / f0 if f0 else float("nan"), f1 > 0.1 * f0)
    if a.gate == "g1":
        if dev.empty or not a.imagenet_dev:
            res["dev_xsrc_vs_imagenet_plus_0.01"] = ("missing dev_log.csv or --imagenet-dev", False)
        else:
            ours = float(dev["dev/cross_source_mAP"].iloc[-1])
            base = json.loads(Path(a.imagenet_dev).read_text())["dev_cross_source_mAP"]
            res["dev_xsrc_vs_imagenet_plus_0.01"] = (f"{ours:.4f} vs {base:.4f}", ours >= base + 0.01)
    else:
        cp = json.loads(Path(a.cellprofiler_dev).read_text())["results"]["cellprofiler"]
        ours = dev.iloc[-1]
        res["dev_xsrc_ge_cellprofiler"] = (
            f"{ours['dev/cross_source_mAP']:.4f} vs {cp['dev_cross_source']['mAP']:.4f}",
            ours["dev/cross_source_mAP"] >= cp["dev_cross_source"]["mAP"],
        )
        res["dev_M1_ge_cellprofiler_minus_0.01"] = (
            f"{ours['dev/M1_mAP']:.4f} vs {cp['M1']['mAP']:.4f}",
            ours["dev/M1_mAP"] >= cp["M1"]["mAP"] - 0.01,
        )
        pr, pc = ours["dev/pos_probe_row"], cp["position_probe"]["bal_acc_row"]
        res["position_probe_le_1.5x_cellprofiler"] = (f"{pr:.4f} vs {pc:.4f}", pr <= 1.5 * pc)
    ok = all(v[1] for v in res.values())
    for k, (v, p) in res.items():
        print(f"{'PASS' if p else 'FAIL'}  {k}: {v}")
    print(f"plan tier at {sps:.0f} samples/s: {tier(sps)}")
    print("GATE", a.gate.upper(), "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

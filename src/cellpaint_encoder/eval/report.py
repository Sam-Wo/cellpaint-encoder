"""Write metric tables (CSV + Markdown) into a results directory."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def _fmt(v):
    if isinstance(v, float):
        return "NaN" if math.isnan(v) else f"{v:.4f}"
    return str(v)


def flat_summary(res: dict) -> dict:
    out = {}
    for sec, d in res.items():
        if isinstance(d, (int, float, str, bool)) and not sec.startswith("_"):
            out[sec] = d
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    out[f"{sec}.{k}"] = v
    return out


def write_report(results: dict[str, dict], out_dir: str | Path, extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = {name: flat_summary(r) for name, r in results.items()}
    df = pd.DataFrame(rows).T
    df.to_csv(out / "summary.csv")
    keys = [
        c
        for c in df.columns
        if c.endswith(
            (
                ".mAP",
                "bal_acc_row",
                "bal_acc_col",
                "edge_auc",
                "bal_acc_source",
                "frac_active_strict",
                "mean_delta",
                "ci_low",
                "ci_high",
            )
        )
        or c == "eps"
    ]
    lines = ["| representation | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, r in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(_fmt(r.get(k)) for k in keys) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps({"results": results, **(extra or {})}, indent=1, default=str), encoding="utf-8"
    )
    return out

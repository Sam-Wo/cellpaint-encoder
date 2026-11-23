"""Statistics (DESIGN.md 4.5): paired per-compound difference, compound bootstrap CI, one-sided
Wilcoxon signed-rank, Holm correction for the secondary family."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_compound_test(
    ap_a: pd.Series, ap_b: pd.Series, n_boot: int = 10000, seed: int = 0, alternative: str = "greater"
) -> dict:
    """ΔAP = AP(a) − AP(b) per compound (index = compound), on compounds present in both."""
    d = pd.concat([ap_a.rename("a"), ap_b.rename("b")], axis=1).dropna()
    delta = (d.a - d.b).to_numpy()
    out = {"n_compounds": len(delta), "mean_delta": float(delta.mean()) if len(delta) else float("nan")}
    if len(delta) < 2:
        out.update(
            ci_low=float("nan"), ci_high=float("nan"), p_wilcoxon=float("nan"), note="fewer than 2 compounds"
        )
        return out
    rng = np.random.default_rng(seed)
    boots = delta[rng.integers(0, len(delta), (n_boot, len(delta)))].mean(1)
    out["ci_low"], out["ci_high"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    out["sd_delta"] = float(delta.std(ddof=1))
    if np.allclose(delta, 0):
        out["p_wilcoxon"] = 1.0
    else:
        out["p_wilcoxon"] = float(wilcoxon(delta, alternative=alternative, zero_method="wilcox").pvalue)
    out["ci_excludes_0"] = bool(out["ci_low"] > 0 or out["ci_high"] < 0)
    return out


def noninferiority(ap_a: pd.Series, ap_b: pd.Series, margin: float = 0.01, **kw) -> dict:
    """P3: lower CI bound of ΔAP > −margin."""
    r = paired_compound_test(ap_a, ap_b, alternative="greater", **kw)
    r["margin"] = margin
    r["noninferior"] = bool(np.isfinite(r["ci_low"]) and r["ci_low"] > -margin)
    return r


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = [k for k, v in pvals.items() if np.isfinite(v)]
    p = np.array([pvals[k] for k in keys])
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    out = {k: float("nan") for k in pvals}
    out.update({k: float(a) for k, a in zip(keys, adj)})
    return out

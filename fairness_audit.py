"""Group fairness diagnostics for a simulated pincode-based socioeconomic proxy."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def fairness_metrics(y_true: Sequence[int], p_default: Sequence[float], group: Sequence[int], risk_cutoff: float) -> dict[str, float]:
    """Compute decision fairness with repayment as the favorable ground-truth outcome.

    `group=1` is the protected/proxy-low-income-area group. Approval means predicted
    default probability is below the cutoff. Equal opportunity is measured among
    borrowers who actually repaid (y=0): approval TPR(group=1)-approval TPR(group=0).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_default, dtype=float)
    g = np.asarray(group, dtype=int)
    approved = p < risk_cutoff

    def rate(mask):
        return float(approved[mask].mean()) if mask.sum() else float("nan")

    ar_protected = rate(g == 1)
    ar_reference = rate(g == 0)
    di = ar_protected / ar_reference if ar_reference > 0 else float("nan")
    good = y == 0
    tpr_protected = rate((g == 1) & good)
    tpr_reference = rate((g == 0) & good)
    eod = tpr_protected - tpr_reference
    return {
        "approval_rate_protected": ar_protected,
        "approval_rate_reference": ar_reference,
        "disparate_impact_ratio": float(di),
        "equal_opportunity_difference": float(eod),
    }


def bias_flag(metrics: dict[str, float], di_floor: float = 0.80, eod_abs_limit: float = 0.10) -> bool:
    di = metrics["disparate_impact_ratio"]
    eod = metrics["equal_opportunity_difference"]
    return bool((np.isfinite(di) and (di < di_floor or di > 1 / di_floor)) or (np.isfinite(eod) and abs(eod) > eod_abs_limit))


def fairness_score(metrics: dict[str, float]) -> float:
    """Lower is better; used only to compare mitigation candidates, not certify fairness."""
    di = metrics["disparate_impact_ratio"]
    eod = metrics["equal_opportunity_difference"]
    di_penalty = abs(np.log(max(di, 1e-6))) if np.isfinite(di) else 10.0
    return float(di_penalty + abs(eod))


def plot_fairness(before: dict[str, float], after: dict[str, float], path: str | Path) -> None:
    labels = ["DI ratio (ideal=1)", "|Equal opportunity diff| (ideal=0)"]
    b = [before["disparate_impact_ratio"], abs(before["equal_opportunity_difference"])]
    a = [after["disparate_impact_ratio"], abs(after["equal_opportunity_difference"])]
    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, b, width, label="Before")
    ax.bar(x + width/2, a, width, label="After proxy removal")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=8)
    ax.legend(); ax.set_title("Fairness audit: before vs mitigation")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)

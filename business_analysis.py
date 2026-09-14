"""Approval/default trade-off and simple rupee impact analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def approval_tradeoff(
    y_true: Sequence[int],
    p_default: Sequence[float],
    avg_loan_size_inr: float = 150_000,
    lgd: float = 0.60,
    good_loan_margin: float = 0.12,
) -> pd.DataFrame:
    """Evaluate cutoffs using realized holdout outcomes.

    loss_avoided_inr values rejected defaults at principal*LGD. good_profit_lost_inr
    values rejected good loans at an explicit 12% contribution-margin assumption; using
    full principal as good loans lost would overstate economic loss.
    """
    y = np.asarray(y_true, dtype=int); p = np.asarray(p_default, dtype=float)
    rows = []
    for cutoff in np.linspace(0.03, 0.60, 58):
        approved = p < cutoff
        rejected = ~approved
        approval_rate = approved.mean()
        default_rate = y[approved].mean() if approved.any() else np.nan
        bad_avoided = int(((y == 1) & rejected).sum())
        good_lost = int(((y == 0) & rejected).sum())
        loss_avoided = bad_avoided * avg_loan_size_inr * lgd
        good_profit_lost = good_lost * avg_loan_size_inr * good_loan_margin
        rows.append({"cutoff": cutoff, "approval_rate": approval_rate,
                     "approved_default_rate": default_rate,
                     "bad_loans_avoided": bad_avoided, "good_loans_lost": good_lost,
                     "loss_avoided_inr": loss_avoided,
                     "good_profit_lost_inr": good_profit_lost,
                     "net_value_inr": loss_avoided - good_profit_lost})
    return pd.DataFrame(rows)


def choose_cutoff(curve: pd.DataFrame, min_approval_rate: float = 0.25) -> pd.Series:
    eligible = curve[curve["approval_rate"] >= min_approval_rate]
    if eligible.empty:
        eligible = curve
    return eligible.loc[eligible["net_value_inr"].idxmax()]


def plot_tradeoff(curve: pd.DataFrame, recommended_cutoff: float, path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(curve["cutoff"], curve["approval_rate"], label="Approval rate")
    ax1.plot(curve["cutoff"], curve["approved_default_rate"], label="Default rate among approved")
    ax1.axvline(recommended_cutoff, linestyle="--", label=f"Recommended cutoff={recommended_cutoff:.2f}")
    ax1.set_xlabel("Default-probability cutoff"); ax1.set_ylabel("Rate")
    ax1.set_title("Approval vs default-rate trade-off"); ax1.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)

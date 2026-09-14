"""Portfolio-level, retrospective risk/return analysis.

This module intentionally evaluates historical score cutoffs in aggregate.
It does not produce applicant-level lending recommendations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def cutoff_analysis(y_true, score, loan_amount, lgd=0.45, cutoffs=None):
    """Estimate portfolio statistics across score cutoffs for research."""
    if cutoffs is None:
        cutoffs = np.arange(0.05, 0.96, 0.05)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score)
    loan = np.asarray(loan_amount, dtype=float)
    rows = []
    for c in cutoffs:
        accepted = s < c
        n = int(accepted.sum())
        exposure = float(loan[accepted].sum())
        defaults = int(y[accepted].sum())
        expected_loss = float((loan[accepted] * y[accepted] * lgd).sum())
        rows.append({
            "risk_cutoff": float(c),
            "portfolio_share": float(n / len(y)),
            "historical_default_rate": float(defaults / n) if n else np.nan,
            "exposure": exposure,
            "expected_loss_at_lgd": expected_loss,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "portfolio_cutoff_analysis.csv", index=False)
    return out


if __name__ == "__main__":
    print("Import cutoff_analysis() after generating retrospective research predictions.")

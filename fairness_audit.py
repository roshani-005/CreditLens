"""Research-only fairness diagnostics for CreditLens.

This module measures group-level disparities in historical default-risk
predictions. It is intentionally not an eligibility or underwriting engine.

The synthetic income-proxy group is included only to demonstrate why proxy
features deserve audit attention in a portfolio analytics setting.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def disparate_impact(y_true, score, group, threshold=0.5):
    """Compare positive prediction rates across groups for diagnostics."""
    pred = (np.asarray(score) >= threshold).astype(int)
    frame = pd.DataFrame({"y": y_true, "pred": pred, "group": group})
    rates = frame.groupby("group")["pred"].mean()
    if len(rates) < 2:
        return np.nan
    return float(rates.min() / rates.max()) if rates.max() else np.nan


def equal_opportunity_difference(y_true, score, group, threshold=0.5):
    """Max-min TPR gap across groups, conditional on actual defaults."""
    pred = (np.asarray(score) >= threshold).astype(int)
    frame = pd.DataFrame({"y": y_true, "pred": pred, "group": group})
    tprs = frame.groupby("group").apply(
        lambda x: recall_score(x["y"], x["pred"], zero_division=0),
        include_groups=False,
    )
    if len(tprs) < 2:
        return np.nan
    return float(tprs.max() - tprs.min())


def audit(y_true, score, group, threshold=0.5):
    """Return transparent group-level diagnostics without a decision output."""
    df = pd.DataFrame({"actual_default": y_true, "score": score, "group": group})
    summary = df.groupby("group").agg(
        population=("score", "size"),
        observed_default_rate=("actual_default", "mean"),
        mean_predicted_risk=("score", "mean"),
    )
    summary["predicted_positive_rate"] = df.assign(
        predicted=(df["score"] >= threshold).astype(int)
    ).groupby("group")["predicted"].mean()
    metrics = {
        "threshold_for_diagnostic_only": threshold,
        "disparate_impact_ratio": disparate_impact(y_true, score, group, threshold),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, score, group, threshold),
    }
    summary.to_csv(OUT / "fairness_group_summary.csv")
    (OUT / "fairness_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return summary, metrics


if __name__ == "__main__":
    print("Import audit() from fairness_audit.py after generating research predictions.")

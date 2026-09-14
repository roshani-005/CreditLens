"""SHAP explanations and lender-friendly reason codes."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model import FittedRiskModel


def _raw_feature_name(transformed: str, raw_features: Iterable[str]) -> str:
    name = transformed.split("__", 1)[-1]
    if name in raw_features:
        return name
    for raw in sorted(raw_features, key=len, reverse=True):
        if name == raw or name.startswith(raw + "_"):
            return raw
    return name


def shap_values(model: FittedRiskModel, X: pd.DataFrame, max_rows: int = 500):
    sample = X[model.feature_columns].head(max_rows).copy()
    Xt = model.transform(sample)
    explainer = shap.TreeExplainer(model.estimator)
    explanation = explainer(Xt)
    names = model.transformed_feature_names()
    explanation.feature_names = list(names)
    return sample, explanation


def save_shap_plots(model: FittedRiskModel, X: pd.DataFrame, output_dir: str | Path = "outputs") -> dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sample, exp = shap_values(model, X)
    raw_names = [_raw_feature_name(n, model.feature_columns) for n in exp.feature_names]
    abs_vals = np.abs(exp.values).mean(axis=0)
    imp = pd.DataFrame({"feature": raw_names, "importance": abs_vals}).groupby("feature", as_index=False).sum().sort_values("importance").tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_title("Global SHAP importance"); ax.set_xlabel("mean |SHAP value|")
    p1 = out / "shap_global_importance.png"
    fig.tight_layout(); fig.savefig(p1, dpi=160, bbox_inches="tight"); plt.close(fig)

    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(exp[0], max_display=12, show=False)
    p2 = out / "shap_individual_example.png"
    plt.tight_layout(); plt.savefig(p2, dpi=160, bbox_inches="tight"); plt.close()
    return {"global": str(p1), "individual": str(p2)}


def _reason_text(feature: str, row: pd.Series) -> str:
    if feature == "obligation_to_income_ratio": return "High total obligations relative to income"
    if feature == "emi_to_income_ratio": return "Requested-loan EMI is high relative to income"
    if feature == "salary_credit_regularity": return "Salary credits appear irregular"
    if feature == "bill_payment_punctuality_score": return "Bill-payment punctuality is weak"
    if feature == "dti": return "Debt-to-income ratio is elevated"
    if feature in {"fico_range_low", "fico_range_high"}: return "Bureau score signal increases estimated risk"
    if feature == "delinq_2yrs": return "Recent delinquency history increases estimated risk"
    if feature == "int_rate": return "Loan pricing is associated with higher observed credit risk"
    if feature == "upi_txn_volume_trend": return "Recent transaction-volume trend increases estimated risk"
    if feature.startswith("upi_txn_frequency"): return "Recent transaction-activity pattern increases estimated risk"
    if feature == "recharge_frequency": return "Mobile recharge pattern increases estimated risk"
    if feature == "annual_inc": return "Income level contributes to the risk estimate"
    if feature == "loan_amnt": return "Requested loan amount contributes to the risk estimate"
    return f"{feature.replace('_', ' ').title()} increases estimated risk"


def reason_codes(model: FittedRiskModel, applicant: pd.DataFrame, top_k: int = 3) -> list[str]:
    """Return top positive SHAP drivers. Positive SHAP means higher default risk."""
    row = applicant[model.feature_columns].iloc[0]
    Xt = model.transform(applicant[model.feature_columns])
    exp = shap.TreeExplainer(model.estimator)(Xt)
    vals = np.asarray(exp.values[0], dtype=float)
    names = model.transformed_feature_names()
    agg: dict[str, float] = {}
    for name, val in zip(names, vals):
        raw = _raw_feature_name(str(name), model.feature_columns)
        agg[raw] = agg.get(raw, 0.0) + float(val)
    ranked = [k for k, v in sorted(agg.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
    if not ranked:
        ranked = [k for k, _ in sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)]
    return [_reason_text(f, row) for f in ranked[:top_k]]

"""Research-only model explainability for CreditLens.

Produces global SHAP importance and plain-English diagnostic reason codes.
These explanations are for model analysis, not individual lending decisions.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

REASON_TEXT = {
    "int_rate": "Interest rate is strongly associated with historical risk.",
    "dti": "Debt-to-income is associated with historical repayment risk.",
    "annual_inc": "Income level contributes to the model's estimated historical risk.",
    "fico_range_low": "Lower historical FICO values are associated with higher risk.",
    "fico_range_high": "FICO range contributes to the model's estimated historical risk.",
    "delinq_2yrs": "Recent delinquency history contributes to the estimated risk.",
    "revol_util": "Revolving-credit utilization contributes to the estimated risk.",
    "emi_to_income_ratio": "EMI burden relative to income contributes to the estimated risk.",
    "obligation_to_income_ratio": "Total simulated obligation burden relative to income contributes to the estimated risk.",
    "salary_credit_regularity": "Salary-credit timing regularity contributes to the estimated risk.",
    "upi_txn_frequency_30d": "Recent UPI activity contributes to the estimated risk.",
    "upi_txn_volume_trend": "Recent transaction-volume trend contributes to the estimated risk.",
    "recharge_frequency": "Mobile-recharge frequency contributes to the estimated risk.",
    "bill_payment_punctuality_score": "Simulated bill-payment punctuality contributes to the estimated risk.",
}


def explain(model_path="models/xgboost.joblib", csv_path="data/processed/creditlens.csv", sample_size=1000):
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"] if isinstance(bundle, dict) and "pipeline" in bundle else bundle
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["default"], errors="ignore")
    X = X.sample(min(sample_size, len(X)), random_state=42)

    pre = pipe.named_steps["preprocess"]
    estimator = pipe.named_steps["model"]
    X_t = pre.transform(X)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_t)
    if isinstance(values, list):
        values = values[-1]
    feature_names = pre.get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT / "shap_global_importance.csv", index=False)

    base_names = []
    for name in importance.head(15)["feature"]:
        raw = name.split("__", 1)[-1]
        raw = raw.split("_", 1)[0] if raw.startswith("cat") else raw
        base_names.append(REASON_TEXT.get(raw, f"{raw} contributes to the model's estimated historical risk."))
    (OUT / "reason_codes.json").write_text(json.dumps(base_names, indent=2))
    return importance


if __name__ == "__main__":
    print("Run explain() after model training and data preparation.")

"""EDA for CreditLens. Every figure is written to outputs/."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def run_eda(df: pd.DataFrame, target: str = "default") -> dict:
    numeric = df.select_dtypes(include="number")
    missing = (df.isna().mean().sort_values(ascending=False) * 100).rename("missing_pct")
    missing.head(20).plot(kind="bar", figsize=(10, 5), title="Top missingness rates")
    plt.ylabel("Missing (%)"); plt.tight_layout(); plt.savefig(OUT / "missingness.png", dpi=160); plt.close()

    df[target].value_counts(normalize=True).sort_index().plot(kind="bar", figsize=(6, 4), title="Default class balance")
    plt.ylabel("Share"); plt.xticks([0, 1], ["Non-default", "Default"], rotation=0)
    plt.tight_layout(); plt.savefig(OUT / "target_imbalance.png", dpi=160); plt.close()

    corr = numeric.corr(numeric_only=True)[target].drop(target).sort_values()
    corr.tail(10).plot(kind="barh", figsize=(8, 5), title="Top positive correlations with default")
    plt.xlabel("Pearson correlation"); plt.tight_layout(); plt.savefig(OUT / "default_correlations.png", dpi=160); plt.close()

    key = [c for c in ["annual_inc", "int_rate", "dti", "obligation_to_income_ratio", "bill_payment_punctuality_score"] if c in df]
    if key:
        df[key].hist(figsize=(12, 8), bins=30); plt.tight_layout(); plt.savefig(OUT / "feature_distributions.png", dpi=160); plt.close()

    return {"rows": len(df), "columns": len(df.columns), "default_rate": float(df[target].mean()), "missing_top": missing.head(10).to_dict()}

"""Data loading, target construction, Indian-fintech feature synthesis, and EDA."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from faker import Faker

from config import BASE_FEATURES, ID_COLUMN, LEAKAGE_COLUMNS, MODEL_FEATURES, OUTPUT_DIR, RANDOM_STATE, TARGET

GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
BAD_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}


def _as_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _emp_length_years(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    out = pd.to_numeric(s.str.extract(r"(\d+)", expand=False), errors="coerce")
    out = out.mask(s.str.contains("< 1", na=False), 0.5)
    return out


def load_lending_club(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load a Kaggle Lending Club CSV and keep only loans with resolved outcomes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Lending Club CSV not found at {path}. Place the Kaggle CSV under data/ "
            "and pass --data data/<filename>. See data/README.md."
        )
    df = pd.read_csv(path, low_memory=False, nrows=max_rows)
    if "loan_status" not in df.columns:
        raise ValueError("Expected a Lending Club 'loan_status' column.")
    resolved = df["loan_status"].isin(GOOD_STATUSES | BAD_STATUSES)
    df = df.loc[resolved].copy()
    df[TARGET] = df["loan_status"].isin(BAD_STATUSES).astype(int)
    return df


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = np.nan
    return out


def clean_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Lending Club fields while avoiding outcome leakage."""
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns], errors="ignore").copy()
    df = _ensure_columns(df, BASE_FEATURES)
    numeric_like = [
        "loan_amnt", "annual_inc", "dti", "int_rate", "installment", "delinq_2yrs",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "mort_acc", "pub_rec_bankruptcies", "fico_range_low", "fico_range_high",
    ]
    for col in numeric_like:
        df[col] = _as_number(df[col])
    df["emp_length"] = _emp_length_years(df["emp_length"])
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)", expand=False)
    df["term"] = pd.to_numeric(df["term"], errors="coerce")
    return df


def add_indian_fintech_layer(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create synthetic Indian-fintech signals without using the target label.

    These variables are behavioral proxies for a portfolio demo, not claims that
    Lending Club borrowers actually had UPI/mobile data. No synthetic feature is
    generated from `default`; doing so would leak the label.

    Real-world rationale:
    - salary_credit_regularity: std-dev of days between simulated monthly salary
      credits. Lower values suggest stable payroll timing/cash-flow predictability.
    - upi_txn_frequency_30d/60d/90d: nested rolling transaction counts. Activity
      can help thin-file lenders observe cash-flow engagement when bureau history is sparse.
    - upi_txn_volume_trend: percentage change in simulated payment volume over 90d;
      abrupt contraction can be a cash-flow stress signal (direction must be validated).
    - recharge_frequency: average days between mobile recharges. It is a weak proxy
      and should never be treated as a protected-trait substitute.
    - bill_payment_punctuality_score: 0-100 score from simulated payment delays;
      higher means more punctual recurring bill behavior.
    - emi_to_income_ratio: monthly installment divided by monthly income.
    - obligation_to_income_ratio: approximate total monthly obligations divided by
      monthly income, combining Lending Club DTI with the new installment burden.
    - pincode_income_proxy/proxy_low_income_area: intentionally included for a
      fairness audit. This is a location-based socioeconomic proxy and is not
      recommended as a production underwriting feature.
    """
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)
    out = clean_base_features(df)
    n = len(out)

    annual_inc = out["annual_inc"].fillna(out["annual_inc"].median()).clip(lower=12_000)
    monthly_income = annual_inc / 12.0
    emp_years = out["emp_length"].fillna(out["emp_length"].median()).clip(0, 40)
    dti = out["dti"].fillna(out["dti"].median()).clip(0, 80)
    installment = out["installment"].fillna(out["installment"].median()).clip(lower=0)
    fico = ((out["fico_range_low"] + out["fico_range_high"]) / 2).fillna(690).clip(500, 850)

    salary_scale = 5.5 - 0.06 * emp_years - 0.006 * (fico - 650)
    out["salary_credit_regularity"] = np.clip(
        np.abs(rng.normal(loc=np.maximum(0.8, salary_scale), scale=1.8, size=n)), 0.2, 15.0
    ).round(2)

    daily_rate = np.clip(0.6 + 0.000015 * monthly_income + rng.gamma(1.8, 0.35, n), 0.2, 12)
    c30 = rng.poisson(daily_rate * 30)
    c60_extra = rng.poisson(daily_rate * 30)
    c90_extra = rng.poisson(daily_rate * 30)
    out["upi_txn_frequency_30d"] = c30
    out["upi_txn_frequency_60d"] = c30 + c60_extra
    out["upi_txn_frequency_90d"] = c30 + c60_extra + c90_extra

    month1 = np.maximum(100.0, monthly_income * rng.uniform(0.15, 0.55, n))
    stress = np.clip((dti - 20) / 120, -0.1, 0.35)
    month3 = np.maximum(50.0, month1 * (1 + rng.normal(0.02 - stress, 0.18, n)))
    out["upi_txn_volume_trend"] = np.clip((month3 - month1) / month1, -0.85, 1.5).round(4)

    out["recharge_frequency"] = np.clip(
        rng.normal(28 - np.log1p(daily_rate) * 2.5, 4.5, n), 7, 60
    ).round(1)

    mean_delay = np.maximum(0, 0.08 * dti + 0.45 * out["salary_credit_regularity"] - 1.0)
    delay_noise = rng.gamma(shape=1.6, scale=1.5, size=n)
    avg_delay_days = np.maximum(0, mean_delay + delay_noise - 2.0)
    out["bill_payment_punctuality_score"] = np.clip(100 - 5.5 * avg_delay_days, 0, 100).round(1)

    out["emi_to_income_ratio"] = np.clip(installment / monthly_income, 0, 3).round(4)
    out["obligation_to_income_ratio"] = np.clip((dti / 100.0) + out["emi_to_income_ratio"], 0, 4).round(4)

    log_income = np.log1p(annual_inc)
    z = (log_income - log_income.mean()) / (log_income.std(ddof=0) + 1e-9)
    proxy = 50 + 14 * z + rng.normal(0, 12, n)
    out["pincode_income_proxy"] = np.clip(proxy, 0, 100).round(1)
    out["proxy_low_income_area"] = (out["pincode_income_proxy"] < 42).astype(int)

    out[ID_COLUMN] = [fake.uuid4() for _ in range(n)]
    return out


def prepare_dataset(path: str | Path, max_rows: int | None = None, seed: int = RANDOM_STATE) -> pd.DataFrame:
    raw = load_lending_club(path, max_rows=max_rows)
    enriched = add_indian_fintech_layer(raw, seed=seed)
    keep = [ID_COLUMN, TARGET] + MODEL_FEATURES
    return _ensure_columns(enriched, keep)[keep].copy()


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: str | Path = OUTPUT_DIR) -> dict[str, str]:
    """Save compact underwriting EDA to /outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    df[TARGET].value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Target imbalance"); ax.set_xlabel("Default (0=good, 1=bad)"); ax.set_ylabel("Share")
    p = output_dir / "eda_target_imbalance.png"; _save_fig(fig, p); paths["target"] = str(p)

    miss = df[MODEL_FEATURES].isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    miss.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top missingness rates"); ax.set_xlabel("Missing share")
    p = output_dir / "eda_missingness.png"; _save_fig(fig, p); paths["missingness"] = str(p)

    numeric = df[MODEL_FEATURES].select_dtypes(include=np.number)
    corr = pd.concat([numeric, df[[TARGET]]], axis=1).corr(numeric_only=True)[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    corr.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Absolute numeric correlation with default"); ax.set_xlabel("|Pearson correlation|")
    p = output_dir / "eda_default_correlation.png"; _save_fig(fig, p); paths["correlation"] = str(p)

    cols = [c for c in ["annual_inc", "dti", "obligation_to_income_ratio", "bill_payment_punctuality_score"] if c in df]
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, max(3, 2.5 * len(cols))))
    if len(cols) == 1: axes = [axes]
    for ax, col in zip(axes, cols):
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if col == "annual_inc" and len(values): values = values.clip(upper=values.quantile(0.99))
        ax.hist(values, bins=35); ax.set_title(col)
    p = output_dir / "eda_feature_distributions.png"; _save_fig(fig, p); paths["distributions"] = str(p)
    return paths


def generate_demo_lending_club(n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a Lending-Club-shaped smoke-test dataset; not a research substitute."""
    rng = np.random.default_rng(seed)
    annual_inc = np.exp(rng.normal(np.log(70_000), 0.55, n)).clip(18_000, 300_000)
    loan_amnt = rng.integers(1_000, 40_000, n)
    dti = np.clip(rng.gamma(3, 6, n), 0, 55)
    fico_mid = np.clip(rng.normal(700, 35, n), 580, 820)
    int_rate = np.clip(6 + (740 - fico_mid) * 0.08 + dti * 0.12 + rng.normal(0, 2, n), 5, 31)
    term = rng.choice([36, 60], n, p=[0.72, 0.28])
    monthly_rate = int_rate / 1200
    installment = loan_amnt * monthly_rate / (1 - (1 + monthly_rate) ** (-term))
    emp = rng.integers(0, 11, n)
    delinq = rng.poisson(np.clip((700 - fico_mid) / 100, 0.02, 2))
    logit = -3.0 + 0.07 * (int_rate - 10) + 0.035 * (dti - 15) + 0.000018 * loan_amnt - 0.009 * (fico_mid - 680)
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, np.clip(p, 0.02, 0.65))
    return pd.DataFrame({
        "loan_status": np.where(y == 1, "Charged Off", "Fully Paid"),
        "loan_amnt": loan_amnt, "annual_inc": annual_inc, "dti": dti,
        "int_rate": int_rate, "installment": installment, "term": [f" {v} months" for v in term],
        "emp_length": [f"{v} years" for v in emp],
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.45, 0.10]),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], n),
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "delinq_2yrs": delinq, "inq_last_6mths": rng.poisson(0.7, n),
        "open_acc": rng.integers(2, 25, n), "pub_rec": rng.poisson(0.15, n),
        "revol_bal": rng.lognormal(9.0, 0.8, n), "revol_util": np.clip(rng.normal(48, 24, n), 0, 120),
        "total_acc": rng.integers(5, 55, n), "mort_acc": rng.poisson(1.3, n),
        "pub_rec_bankruptcies": rng.binomial(1, 0.06, n),
        "fico_range_low": fico_mid - 2, "fico_range_high": fico_mid + 2,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to Kaggle Lending Club CSV")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--demo", action="store_true", help="Write smoke-test demo data instead")
    parser.add_argument("--out", type=str, default="data/processed.csv")
    args = parser.parse_args()
    if args.demo:
        raw = generate_demo_lending_club()
        raw.to_csv("data/demo_lending_club.csv", index=False)
        df = add_indian_fintech_layer(raw)
        df[TARGET] = raw["loan_status"].eq("Charged Off").astype(int).to_numpy()
        df = df[[ID_COLUMN, TARGET] + MODEL_FEATURES]
    else:
        if not args.data: parser.error("--data is required unless --demo is used")
        df = prepare_dataset(args.data, max_rows=args.max_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    run_eda(df)
    print(f"Prepared {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()

"""CreditLens data preparation.

Input: Lending Club loan-level CSV downloaded from Kaggle.
Output: a modelling table with original credit variables plus a deterministic
synthetic Indian-fintech behavioral layer. Synthetic variables are clearly
marked and must not be interpreted as observed borrower data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
TARGET = "default"
SYNTHETIC_FEATURES = [
    "salary_credit_regularity", "upi_txn_frequency_30d", "upi_txn_frequency_60d",
    "upi_txn_frequency_90d", "upi_txn_volume_trend", "recharge_frequency",
    "bill_payment_punctuality_score", "emi_to_income_ratio", "obligation_to_income_ratio",
    "pincode_income_proxy", "income_proxy_group"
]


def _first_existing(df, names, default=np.nan):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def load_lending_club(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    status = df.get("loan_status")
    if status is None:
        raise ValueError("Expected a Lending Club CSV containing loan_status.")
    keep = status.isin(["Fully Paid", "Charged Off", "Default"])
    df = df.loc[keep].copy()
    df[TARGET] = status.loc[keep].isin(["Charged Off", "Default"]).astype(int)

    # Use stable, widely available Lending Club fields and drop post-outcome fields.
    fields = [
        "loan_amnt", "annual_inc", "int_rate", "installment", "dti",
        "delinq_2yrs", "open_acc", "pub_rec", "revol_util", "fico_range_low",
        "fico_range_high", "emp_length", "home_ownership", "purpose", "term"
    ]
    available = [c for c in fields if c in df.columns]
    out = df[available + [TARGET]].copy()
    return out


def _numeric(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)


def add_indian_fintech_layer(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add synthetic transaction/banking behavior on top of each loan row.

    Rationale:
    * salary_credit_regularity: standard deviation of simulated days between
      monthly salary credits; stable payroll timing is a cash-flow consistency signal.
    * UPI frequencies: recent transaction counts approximate account activity and
      liquidity engagement over 30/60/90 day windows.
    * upi_txn_volume_trend: change from an earlier 45-day period to the recent
      45-day period, useful for detecting deteriorating cash-flow activity.
    * recharge_frequency: average days between simulated mobile recharges, a proxy
      for recurring digital-payment behavior rather than creditworthiness itself.
    * bill_payment_punctuality_score: 0-100 summary of simulated payment delays.
    * emi_to_income_ratio and obligation_to_income_ratio: affordability measures.
    * pincode_income_proxy: synthetic area-level income proxy used only for fairness
      auditing. It is deliberately treated as a proxy, not as protected identity data.
    """
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)
    n = len(df)
    income = _numeric(_first_existing(df, ["annual_inc"]), 600000).clip(lower=120000)
    loan = _numeric(_first_existing(df, ["loan_amnt"]), 150000).clip(lower=10000)
    installment = _numeric(_first_existing(df, ["installment"]), loan / 24).clip(lower=500)
    dti = _numeric(_first_existing(df, ["dti"]), 18).clip(0, 80)

    # Simulate six monthly salary-credit intervals per applicant, then take std dev.
    salary_noise = np.clip(rng.lognormal(mean=-0.7, sigma=0.55, size=(n, 6)), 0.2, 6)
    salary_gaps = np.clip(30 + rng.normal(0, 1.3, size=(n, 6)) * salary_noise, 20, 45)
    out = df.copy()
    out["salary_credit_regularity"] = salary_gaps.std(axis=1).round(2)

    monthly_income = income / 12
    activity_base = np.clip(8 + 0.000015 * monthly_income - 0.12 * dti, 2, 65)
    out["upi_txn_frequency_30d"] = rng.poisson(activity_base).astype(float)
    out["upi_txn_frequency_60d"] = out["upi_txn_frequency_30d"] + rng.poisson(activity_base * 0.9)
    out["upi_txn_frequency_90d"] = out["upi_txn_frequency_60d"] + rng.poisson(activity_base * 0.85)

    # Recent-vs-prior transaction volume trend. Noise makes the synthetic layer non-deterministic per borrower while the seed keeps runs reproducible.
    recent = rng.lognormal(np.log(np.maximum(monthly_income * 0.65, 10000)), 0.35)
    prior = rng.lognormal(np.log(np.maximum(monthly_income * 0.70, 10000)), 0.35)
    out["upi_txn_volume_trend"] = ((recent - prior) / np.maximum(prior, 1) * 100).clip(-90, 250).round(2)

    recharge_days = np.clip(rng.normal(18 + 0.10 * dti, 4.5, n), 5, 45)
    out["recharge_frequency"] = recharge_days.round(2)

    delay_days = np.clip(rng.gamma(shape=1.8, scale=1.2, size=n) + 0.06 * dti, 0, 18)
    out["bill_payment_punctuality_score"] = np.clip(100 - delay_days * 5.5, 0, 100).round(2)

    out["emi_to_income_ratio"] = (installment / monthly_income).clip(0, 2).round(4)
    out["obligation_to_income_ratio"] = (out["emi_to_income_ratio"] + dti / 100 * 0.55 + rng.normal(0, .015, n)).clip(0, 2).round(4)

    # Faker provides realistic-looking pincode strings without tying rows to real households.
    out["pincode"] = [fake.postcode() for _ in range(n)]
    area_band = rng.choice(["lower", "middle", "upper"], size=n, p=[0.35, 0.50, 0.15])
    out["income_proxy_group"] = area_band
    out["pincode_income_proxy"] = pd.Series(area_band, index=out.index).map({"lower": 0.75, "middle": 1.0, "upper": 1.30}).astype(float)

    # Remove the synthetic pincode itself from modelling; retain only the proxy used in the fairness experiment.
    out = out.drop(columns=["pincode"])
    return out


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in [TARGET, "income_proxy_group"]:
            continue
        if out[c].dtype == "object":
            out[c] = out[c].fillna("Unknown").astype(str)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Lending Club emp_length is categorical text such as "10+ years" and is kept categorical.
    return out


def prepare(input_csv: str, output_csv: str, sample: int | None = None) -> pd.DataFrame:
    df = load_lending_club(input_csv)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=SEED).reset_index(drop=True)
    df = add_indian_fintech_layer(df)
    df = clean_and_encode(df)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/lending_club.csv")
    parser.add_argument("--output", default="data/processed/creditlens.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    result = prepare(args.input, args.output, args.sample)
    print(f"Saved {len(result):,} rows to {args.output}")

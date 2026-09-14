from pathlib import Path

RANDOM_STATE = 42
TARGET = "default"
ID_COLUMN = "synthetic_customer_id"
PROTECTED_PROXY = "proxy_low_income_area"
OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")

# Deliberately conservative feature set. Post-origination fields are excluded because
# they leak future repayment information into an underwriting decision.
LEAKAGE_COLUMNS = {
    "recoveries", "collection_recovery_fee", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "out_prncp", "out_prncp_inv", "last_credit_pull_d", "hardship_flag",
    "debt_settlement_flag", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
}

BASE_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "int_rate", "installment", "term",
    "emp_length", "home_ownership", "purpose", "verification_status",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "fico_range_low", "fico_range_high",
]

SYNTHETIC_FEATURES = [
    "salary_credit_regularity",
    "upi_txn_frequency_30d",
    "upi_txn_frequency_60d",
    "upi_txn_frequency_90d",
    "upi_txn_volume_trend",
    "recharge_frequency",
    "bill_payment_punctuality_score",
    "emi_to_income_ratio",
    "obligation_to_income_ratio",
    "pincode_income_proxy",
    PROTECTED_PROXY,
]

MODEL_FEATURES = BASE_FEATURES + SYNTHETIC_FEATURES

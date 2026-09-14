"""Streamlit score-explanation demo for fictional/synthetic applicants only."""
from pathlib import Path
import joblib, pandas as pd, streamlit as st
from explainability import reason_codes

MODEL_PATH = Path("artifacts/creditlens_model.joblib")
st.set_page_config(page_title="CreditLens Research Demo", page_icon="🔎", layout="centered")
st.title("CreditLens — Research Demo")
st.info("Use fictional or synthetic inputs only. This demo does not make real loan eligibility decisions.")
if not MODEL_PATH.exists():
    st.error("Model artifact not found. Run `python run_pipeline.py --demo` or train on your local research dataset first."); st.stop()

bundle = joblib.load(MODEL_PATH)
model, calibrator = bundle["model"], bundle["calibrator"]
defaults, features = dict(bundle["feature_defaults"]), list(bundle["feature_columns"])

c1, c2 = st.columns(2)
with c1:
    loan_amnt = st.number_input("Fictional loan amount", min_value=1000.0, value=float(defaults.get("loan_amnt", 15000)), step=1000.0)
    annual_inc = st.number_input("Fictional annual income", min_value=10000.0, value=float(defaults.get("annual_inc", 60000)), step=5000.0)
    dti = st.number_input("Debt-to-income (%)", min_value=0.0, max_value=100.0, value=float(defaults.get("dti", 18.0)))
    installment = st.number_input("Monthly installment", min_value=0.0, value=float(defaults.get("installment", 400.0)), step=50.0)
    salary_reg = st.number_input("Salary timing variability (days)", min_value=0.0, max_value=30.0, value=float(defaults.get("salary_credit_regularity", 3.0)))
with c2:
    bill = st.slider("Bill punctuality score", 0, 100, int(defaults.get("bill_payment_punctuality_score", 85)))
    upi30 = st.number_input("UPI count — 30d", min_value=0, value=int(defaults.get("upi_txn_frequency_30d", 45)))
    upi60 = st.number_input("UPI count — 60d", min_value=0, value=int(defaults.get("upi_txn_frequency_60d", 90)))
    upi90 = st.number_input("UPI count — 90d", min_value=0, value=int(defaults.get("upi_txn_frequency_90d", 135)))
    trend = st.number_input("UPI volume trend (decimal)", min_value=-1.0, max_value=2.0, value=float(defaults.get("upi_txn_volume_trend", 0.0)), step=.05)

row = defaults.copy(); monthly_income = max(annual_inc / 12, 1.0)
row.update({"loan_amnt":loan_amnt, "annual_inc":annual_inc, "dti":dti, "installment":installment,
            "salary_credit_regularity":salary_reg, "bill_payment_punctuality_score":bill,
            "upi_txn_frequency_30d":upi30, "upi_txn_frequency_60d":max(upi60, upi30),
            "upi_txn_frequency_90d":max(upi90, upi60, upi30), "upi_txn_volume_trend":trend,
            "emi_to_income_ratio":installment/monthly_income,
            "obligation_to_income_ratio":dti/100 + installment/monthly_income})
applicant = pd.DataFrame([row])[features]

if st.button("Explain synthetic score", type="primary"):
    raw = float(model.predict_proba(applicant)[0]); risk = float(calibrator.predict([raw])[0])
    st.metric("Calibrated model PD (research output)", f"{risk:.1%}")
    st.caption("This is a model-behavior demonstration, not an approve/reject recommendation.")
    st.subheader("Top SHAP reason codes")
    try:
        for reason in reason_codes(model, applicant, 3): st.write(f"• {reason}")
    except Exception:
        st.write("Reason-code generation unavailable in this environment; inspect the saved SHAP plots.")

# CreditLens: A Bias-Aware Credit Risk Analytics Engine

> **Research / portfolio project:** CreditLens studies historical default-risk prediction, probability calibration, reject-inference methodology, fairness diagnostics, and explainability in an Indian fintech-style dataset. It is not an operational lending or applicant-eligibility system.

## 1. Problem

Thin-file borrowers can have limited traditional bureau history even when they generate useful cash-flow signals through salary credits, UPI activity, recharges, and bill payments. CreditLens asks a research question: **can these additional behavioral signals improve historical default-risk estimation without hiding fairness or calibration problems?**

## 2. Data

The starting point is the Lending Club historical loan dataset. `data_prep.py` adds a clearly labeled synthetic Indian-fintech feature layer because the public dataset does not contain real UPI, recharge, or bill-payment records.

Synthetic features and rationale:

- `salary_credit_regularity`: variability in simulated monthly salary-credit intervals; more regular cash-flow timing can represent income stability.
- `upi_txn_frequency_30d/60d/90d`: simulated rolling transaction counts; represent recent digital-activity intensity.
- `upi_txn_volume_trend`: percentage change between recent and prior simulated transaction volume; represents directional change in activity.
- `recharge_frequency`: average simulated days between mobile recharges; represents recurring digital-payment behavior.
- `bill_payment_punctuality_score`: 0–100 score derived from simulated payment delays; represents consistency in bill servicing.
- `emi_to_income_ratio`: simulated installment burden relative to income.
- `obligation_to_income_ratio`: simulated total-obligation burden relative to income.
- `pincode_income_proxy`: intentionally synthetic socioeconomic proxy used only for fairness-audit demonstration.

These are **not real Indian borrower records** and should never be presented as such.

## 3. Pipeline

1. `data_prep.py` — clean the Lending Club target and add the synthetic behavioral layer.
2. `eda.py` — inspect missingness, target imbalance, distributions, and correlations.
3. `model.py` — compare interpretable Logistic Regression with XGBoost and calibrate the XGBoost probabilities on a held-out validation set.
4. `fairness_audit.py` — calculate group-level disparate-impact and equal-opportunity diagnostics.
5. `explainability.py` — generate SHAP global importance and plain-English diagnostic reason codes.
6. `business_analysis.py` — evaluate aggregate historical score-cutoff trade-offs using exposure and expected-loss calculations.

## 4. Why Logistic Regression first?

Logistic Regression is a useful baseline in credit-risk work because its coefficients have a direct directional interpretation after preprocessing, its behavior is comparatively transparent, and probability outputs are straightforward to monitor. A tree ensemble can capture nonlinear interactions that a linear model misses, so XGBoost is used as the stronger research benchmark.

## 5. Why class weights instead of SMOTE?

SMOTE can interpolate synthetic borrowers between minority-class observations. For credit data, that can create combinations of income, utilization, obligations, and behavioral variables that are not economically plausible. It can also alter probability estimates. CreditLens therefore uses class weighting while leaving the observed feature space intact.

## 6. Calibration

The model compares raw and calibrated probabilities using a calibration curve and Brier score. Calibration matters because a predicted probability should be interpreted as a probability, not merely as a ranking score. Better-calibrated probabilities are useful for portfolio analytics and scenario analysis.

## 7. Reject inference — research boundary

A production reject-inference system can influence who receives credit, so CreditLens does **not** implement an applicant-level automated approval/rejection engine. The repository is intended to discuss reject-inference concepts and evaluate them retrospectively on synthetic or historical research data. Any parceling/fuzzy-label experiment should be treated as methodological research, with its assumptions and uncertainty reported rather than converted into an operational eligibility rule.

## 8. Fairness

The project intentionally creates a synthetic `pincode_income_proxy` to demonstrate proxy-risk auditing. `fairness_audit.py` reports:

- **Disparate impact ratio:** ratio of the lowest to highest group positive-prediction rate.
- **Equal opportunity difference:** max-min difference in true-positive rates across groups.

If a proxy materially changes model behavior, the appropriate research response is to compare models with and without the proxy and document the trade-off. A fairness metric alone does not prove that a model is fair.

## 9. Explainability

SHAP is used for global feature importance. The project maps important features into plain-English diagnostic statements so that model behavior can be inspected without treating a model explanation as a causal claim.

## 10. Portfolio analysis

`business_analysis.py` evaluates historical outcomes across score thresholds at the **portfolio level**. It reports population share, historical default rate, exposure, and expected loss under an assumed LGD. These are scenario-analysis outputs, not applicant-level decisions.

## 11. Reproducibility

```bash
pip install -r requirements.txt
python data_prep.py --input data/raw/lending_club.csv --output data/processed/creditlens.csv
python eda.py
python model.py
```

Then run the research diagnostics after model outputs exist:

```bash
python fairness_audit.py
python explainability.py
python business_analysis.py
```

The exact Lending Club file is intentionally not committed because of dataset size/licensing and reproducibility concerns. Place the downloaded CSV at `data/raw/lending_club.csv`.

## 12. Limitations

- Lending Club is not an Indian fintech population; the Indian behavioral layer is synthetic.
- Synthetic behavioral variables can demonstrate pipeline design but cannot establish real-world predictive value.
- Historical Lending Club labels reflect historical underwriting and economic conditions.
- Fairness metrics depend on the selected groups, outcome definition, and threshold.
- SHAP describes model attribution, not causality.
- Reject inference is especially assumption-sensitive because rejected applicants lack observed repayment outcomes.
- This repository is for research and portfolio analysis, not live credit underwriting.

## Repository structure

```text
CreditLens/
├── data/raw/                 # place source dataset here locally
├── data/processed/           # generated modelling table
├── models/                   # generated model artifacts
├── outputs/                  # generated metrics and plots
├── data_prep.py
├── eda.py
├── model.py
├── fairness_audit.py
├── explainability.py
├── business_analysis.py
├── requirements.txt
└── README.md
```

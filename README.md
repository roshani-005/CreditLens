# CreditLens: A Bias-Aware Underwriting Engine for Thin-File Borrowers

CreditLens is an end-to-end credit-risk portfolio project for an **Indian fintech personal-lending context**, built around three problems that are often skipped in demo models: thin-file feature engineering, approved-only selection bias (reject inference), and group fairness. The model starts from Lending Club loan outcomes, adds a clearly labeled synthetic Indian digital-behavior layer, calibrates probability of default (PD), audits a socioeconomic location proxy, generates SHAP reason codes, and exposes the final score through Streamlit.

> **Important:** Lending Club is a U.S. dataset and the UPI/mobile/pincode layer is synthetic. This repository is a modeling prototype, not a validated Indian underwriting system or a basis for real lending decisions.

## Problem statement

Thin-file borrowers may have too little bureau history for a conventional scorecard, but a lender may observe permissioned cash-flow and payment-behavior signals. A useful underwriting model therefore needs more than ROC-AUC: it needs calibrated PDs, selection-bias awareness, explainability, fairness monitoring, and a decision threshold tied to economics.

CreditLens predicts `default=1` for resolved Lending Club outcomes (`Charged Off`/`Default`) and `default=0` for `Fully Paid`. It deliberately excludes post-origination fields such as recoveries and last-payment information to reduce target leakage.

## Data and synthetic Indian-fintech layer

Place a Kaggle Lending Club CSV in `data/` (the raw dataset is not redistributed here). `data_prep.py` normalizes common Lending Club fields and creates these synthetic application-time signals **without reading the target label**:

| Feature | Construction | Real-world rationale |
|---|---|---|
| `salary_credit_regularity` | Std-dev-like timing variability in days | Stable salary timing can proxy cash-flow predictability |
| `upi_txn_frequency_30d/60d/90d` | Nested simulated rolling transaction counts | Digital activity can add behavioral depth for thin-file applicants |
| `upi_txn_volume_trend` | Simulated 90-day volume % change | Contraction may signal stress; direction must be validated empirically |
| `recharge_frequency` | Avg simulated days between recharges | Weak behavioral signal; should be treated cautiously |
| `bill_payment_punctuality_score` | 0–100 from simulated payment delays | Recurring-payment discipline signal |
| `emi_to_income_ratio` | New EMI / monthly income | Affordability of requested loan |
| `obligation_to_income_ratio` | Existing DTI + new EMI burden | Approximate total affordability pressure |
| `pincode_income_proxy` | Noisy income-linked area index | **Intentionally risky proxy used for the fairness audit** |
| `proxy_low_income_area` | Binary low-area-income group | Audit group only, not a protected identity claim |

`Faker(en_IN)` is used only to create synthetic record IDs. No fabricated identity data enters the model.

## Modeling approach

The pipeline in `run_pipeline.py` performs EDA, a logistic-regression baseline, XGBoost, reject inference, calibration, fairness mitigation, SHAP explanations, and business threshold selection.

**Why logistic regression still matters in lending.** It is easy to audit, coefficients can be reviewed, behavior is stable, and model-risk/governance teams can trace how inputs affect score direction. CreditLens keeps it as a transparent benchmark even though XGBoost is the main nonlinear model.

**Class imbalance.** Both models use class/sample weighting. CreditLens intentionally avoids SMOTE because interpolated tabular borrowers can contain impossible financial combinations and can distort the empirical score distribution used for PD calibration.

**Reject inference.** A simulated legacy policy approves roughly 70% of the training population from FICO, DTI, income, and delinquency signals. Labels for the remainder are treated as hidden. An approved-only XGBoost first predicts PD for rejects; fuzzy augmentation then duplicates each rejected applicant into good/bad parcels weighted by `(1-p_bad)` and `p_bad`. This preserves uncertainty instead of turning model guesses into certain pseudo-labels.

**Calibration.** Platt scaling and isotonic regression are compared on a calibration split containing only historically approved observations. Calibration is selected only if it lowers validation Brier score. This matters because risk-based pricing, expected-loss calculations, and cutoff economics consume the *probability level*, not just ranking quality.

**Fairness.** The audit treats `proxy_low_income_area=1` as the monitored group and reports:

- disparate impact ratio = protected-group approval rate / reference-group approval rate;
- equal opportunity difference = approval rate among actual good borrowers in the monitored group minus the same rate in the reference group.

If thresholds flag a disparity, the pipeline retrains after removing both pincode proxy fields. It deploys that mitigation only when the fairness composite improves and ROC-AUC falls by no more than 2 percentage points. The before/after metrics are always reported rather than declaring the system “fair.”

**Explainability.** SHAP produces global importance plus an individual waterfall. Positive per-applicant SHAP contributions are aggregated back from one-hot columns to raw features and translated into plain-English reason codes such as “High total obligations relative to income” or “Salary credits appear irregular.”

## Business layer

The final calibrated PD is swept across score cutoffs. For every cutoff CreditLens calculates approval rate, default rate among approved loans, bad loans avoided, good loans lost, and a simple rupee value estimate. The default assumptions are:

- average loan size: **₹150,000**;
- loss given default (LGD): **60%**;
- contribution margin on a good loan: **12%**.

The recommended cutoff maximizes `loss avoided - good-loan contribution margin lost`, subject to at least a 25% approval rate. These assumptions are intentionally explicit so they can be replaced with a lender's unit economics.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Real experiment
python run_pipeline.py --data data/<lending_club_file>.csv

# Code-path smoke test only — not empirical results
python run_pipeline.py --demo

# App (after training)
streamlit run app.py
```

The trained bundle is written to `artifacts/creditlens_model.joblib`. Generated plots and tables are written to `/outputs` (see `outputs/README.md`).

## What did not work / what this project deliberately avoids

1. **SMOTE:** rejected because synthetic interpolation can create financially inconsistent applicants and damage calibration.
2. **Hard pseudo-labeling of rejects:** rejected because a 0.51 PD should not become a certain “default.” Fuzzy augmentation retains uncertainty through weights.
3. **Using post-loan outcome fields:** rejected as leakage. Recoveries, last-payment fields, settlement information, and similar columns are excluded.
4. **Treating pincode as a harmless feature:** rejected. It is explicitly stress-tested as a socioeconomic proxy and removed when the audit/utility rule supports mitigation.
5. **Claiming Indian validity from Lending Club:** rejected. The Indian fintech variables are synthetic and must be replaced by consented, governed local data before any real validation.

## Results

A real `metrics.json` is generated only after running on the actual Lending Club CSV. The repository does **not** hard-code or invent performance numbers. The `--demo` mode exists to verify plumbing; its results are synthetic and should never be quoted as model evidence.

The metrics artifact reports, side by side: logistic baseline performance, approved-only XGBoost, post-reject-inference XGBoost, Brier score before/after calibration, fairness before/after proxy removal, whether mitigation was deployed, and the recommended business cutoff with ₹ impact assumptions.

## Repository structure

```text
CreditLens/
├── data_prep.py              # Lending Club loader, synthetic layer, EDA
├── model.py                  # Logistic/XGBoost, class weighting, calibration
├── reject_inference.py       # Legacy approval simulation + fuzzy augmentation
├── fairness_audit.py         # DI, equal opportunity, mitigation comparison
├── explainability.py         # SHAP plots + reason codes
├── business_analysis.py      # Cutoff curve + ₹ economics
├── run_pipeline.py           # End-to-end orchestrator
├── app.py                    # Streamlit underwriting UI
├── config.py                 # Feature/leakage configuration
├── tests/                    # Reproducibility and logic tests
├── outputs/                  # Generated plots/tables
├── artifacts/                # Trained model bundle (git-ignored)
└── data/                     # Local Kaggle CSV (git-ignored)
```

## Limitations and production risks

This project does not solve consent, data provenance, adverse-action compliance, model governance, drift, fraud, identity, bureau matching, pricing policy, or legal eligibility. Synthetic digital features can encode socioeconomic differences; even after proxy removal, correlated variables can preserve disparities. Reject inference is assumption-dependent and cannot recover true labels for historical rejects. Calibration can drift as applicant mix and macro conditions change. A production system would need time-based validation, challenger monitoring, stability/drift tests, local-outcome data, security controls, human-review policy, and independent legal/model-risk review.

## Tests

```bash
pytest -q
```

The tests check deterministic synthetic-feature behavior, nested UPI windows, target independence, fairness formulas, and reject-inference training.

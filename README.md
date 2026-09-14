# CreditLens: A Bias-Aware Underwriting Engine for Thin-File Borrowers

CreditLens is a portfolio-grade research prototype for **credit-risk scoring in an Indian fintech lending context**, with a focus on thin-file and underserved borrowers. It combines interpretable credit modeling, gradient boosting, reject inference, probability calibration, fairness auditing, SHAP explanations, and a business cutoff layer.

> **Important:** This is an educational/research prototype, not a production lending policy. The Indian transaction layer is synthetic and is not evidence that UPI or other alternative-data features should be used for real credit decisions.

## Problem

Traditional underwriting can be difficult for thin-file borrowers because conventional bureau variables may contain limited information. CreditLens explores whether a richer feature layer—while explicitly testing bias and calibration—can support more informed risk segmentation.

## Data

The project is designed to start from the **Lending Club Kaggle dataset**: https://www.kaggle.com/datasets/wordsforthewise/lending-club

The raw Kaggle files are intentionally **not committed** because they are large. Run:

```bash
python scripts/download_data.py
```

Then prepare the data:

```bash
python -m src.data_prep --input data/raw/accepted_2007_to_2018Q4.csv.gz --output data/processed/creditlens_features.csv
```

### Synthetic Indian-fintech feature layer

The numeric layer is generated with NumPy on top of Lending Club records. These features are synthetic proxies for behaviors a fintech lender might observe:

| Feature | Rationale |
|---|---|
| `salary_credit_regularity` | Captures stability of the interval between recurring salary credits; lower variation can indicate steadier cash-flow timing. |
| `upi_txn_frequency_30d/60d/90d` | Represents recent transaction activity and provides short-, medium-, and longer-window behavioral signals. |
| `upi_txn_volume_trend` | Captures whether transaction volume is increasing or decreasing over the simulated 90-day period. |
| `recharge_frequency` | Uses mobile-recharge cadence as a synthetic proxy for recurring digital-payment behavior. |
| `bill_payment_punctuality_score` | Summarizes simulated bill-payment delays into a 0–100 punctuality score. |
| `emi_to_income_ratio` | Measures the modeled EMI burden relative to annual income; higher burden can imply less repayment headroom. |
| `obligation_to_income_ratio` | Adds existing-obligation burden to the modeled EMI burden. |
| `pincode_income_proxy` | Synthetic socioeconomic proxy used specifically for the fairness audit; it is intentionally treated as sensitive/proxy-like. |

The feature-generation code contains comments documenting these assumptions.

## Modeling pipeline

1. **EDA** — target imbalance, missingness, distributions, and default correlations.
2. **Baseline** — Logistic Regression with class weights. Logistic Regression remains important in lending because its coefficients and score contributions are easier to interpret, validate, document, and govern.
3. **Main model** — XGBoost classifier with class weighting.
4. **Class imbalance** — handled with class weights rather than SMOTE. SMOTE can interpolate between financial records and create unrealistic combinations (for example, synthetic income/debt/credit-history combinations), while also complicating probability calibration and governance.
5. **Reject inference** — simulates an approved-only historical sample, predicts the hidden rejected pool, assigns soft labels using a fuzzy/parceling approach, and retrains with weighted pseudo-observations.
6. **Calibration** — isotonic regression is used when calibration improves probability quality; Brier scores and calibration curves are reported.
7. **Fairness audit** — disparate impact ratio and equal opportunity difference are calculated across a synthetic pincode-income proxy. Proxy removal is tested as a mitigation and reported honestly even if it does not improve every metric.
8. **Explainability** — SHAP produces global importance and applicant-level reason codes.
9. **Business layer** — score-cutoff curves estimate approval/default trade-offs and a simple INR impact using an assumed average loan size and LGD.
10. **Deployment** — Streamlit app returns calibrated PD, reason codes, and an approve/manual-review/reject recommendation.

## Running the demo without Kaggle data

The repository includes a deterministic Lending Club-shaped demo generator so the pipeline can be tested without downloading the large source dataset.

```bash
pip install -r requirements.txt
python scripts/run_demo.py
streamlit run app.py
```

The demo generates plots and CSV/JSON artifacts under `outputs/` and model artifacts under `models/`. These generated artifacts are ignored by Git by default.

## Real-data training

After downloading the Kaggle file and preparing it, run:

```bash
python scripts/train_real.py --input data/raw/accepted_2007_to_2018Q4.csv.gz
```

This creates the processed feature table. The modular pipeline functions in `src/pipeline.py` can then be used to run the modeling stages on the prepared data.

## Reject inference: why it matters

A lender historically observes outcomes mainly for people it approved. Rejected applicants therefore have missing outcomes, creating **selection bias**. CreditLens simulates this problem by applying an old approval cutoff, hiding outcomes for the rejected pool, estimating soft default probabilities, and retraining with fuzzy labels. The evaluation keeps a separate validation sample so the reject-inference comparison is not presented as if pseudo-labels were ground truth.

## Calibration and risk-based pricing

A ranking model can identify risky borrowers without producing probabilities that are numerically trustworthy. Calibration makes a predicted PD closer to the observed default frequency for applicants with similar scores. This matters when pricing risk, setting provisions, defining approval bands, or estimating expected loss.

## Fairness

The project deliberately uses a synthetic `pincode_income_proxy` only as an audit variable. It reports:

- **Disparate impact ratio:** approval rate of the lower-proxy group divided by the higher-proxy group.
- **Equal opportunity difference:** true-positive-rate difference across the two groups.

A simple proxy-removal mitigation is evaluated. If the metric gets worse, the README/results do **not** hide that outcome; mitigation is a hypothesis to test, not a guaranteed fix.

## What did not work / limitations

- The synthetic Indian feature layer is illustrative; it is not real UPI/bank/recharge data.
- Lending Club is a US historical dataset, so its population and product context do not represent Indian fintech borrowers.
- XGBoost is the main model by design, but on the deterministic demo fixture Logistic Regression happens to achieve higher ROC-AUC. This is retained rather than cherry-picked away.
- Proxy removal alone does not guarantee fairness improvement; the demo audit shows why mitigation must be measured rather than assumed.
- The demo business cutoff is illustrative and should not be interpreted as a production underwriting threshold.
- Reject inference relies on simulated hidden labels and fuzzy assumptions; real deployment requires careful reject-inference governance and monitoring.
- Production use would require validation for drift, stability, adverse impact, privacy, consent, explainability, model risk, and applicable lending regulation.

## Project structure

```text
CreditLens/
├── app.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_data.py
│   ├── run_demo.py
│   └── train_real.py
├── src/
│   ├── data_prep.py
│   ├── eda.py
│   ├── model.py
│   ├── reject_inference.py
│   ├── fairness_audit.py
│   ├── business_layer.py
│   └── pipeline.py
├── tests/
│   └── test_core.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── outputs/
```

## Resume-ready positioning

**CreditLens — Bias-Aware Credit Risk Scoring Engine**  
Built an end-to-end credit-risk pipeline combining Logistic Regression/XGBoost, class-weighted learning, fuzzy reject inference, probability calibration, fairness auditing, SHAP reason codes, and cutoff-based expected-loss analysis for thin-file lending.

## Disclaimer

CreditLens is a portfolio/research project. It should not be used to make real lending decisions without appropriate legal, compliance, model-risk, privacy, fairness, and credit-policy review.

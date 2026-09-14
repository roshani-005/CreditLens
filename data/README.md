# Data

CreditLens expects a **Lending Club loan-level CSV from Kaggle** containing `loan_status` and common application-time fields such as `loan_amnt`, `annual_inc`, `dti`, `int_rate`, `installment`, `fico_range_low`, and `fico_range_high`.

The raw Kaggle file is intentionally not committed to this repository. Download the dataset under its applicable Kaggle terms, place the CSV in this directory, then run:

```bash
python run_pipeline.py --data data/<your_lending_club_file>.csv
```

For a code-path smoke test only, use:

```bash
python run_pipeline.py --demo
```

The demo generator creates Lending-Club-shaped synthetic rows. **Its metrics are not evidence of real underwriting performance.**

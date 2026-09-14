# Outputs

Running `python run_pipeline.py --data ...` writes the following reproducible artifacts here:

- `eda_target_imbalance.png`
- `eda_missingness.png`
- `eda_default_correlation.png`
- `eda_feature_distributions.png`
- `reject_inference_population_shift.png`
- `reject_inference_comparison.csv`
- `calibration_curve.png`
- `fairness_before_after.png`
- `fairness_metrics.csv`
- `shap_global_importance.png`
- `shap_individual_example.png`
- `approval_default_tradeoff.png`
- `approval_default_tradeoff.csv`
- `metrics.json`

Generated outputs are git-ignored by default to avoid presenting demo/synthetic results as empirical findings.

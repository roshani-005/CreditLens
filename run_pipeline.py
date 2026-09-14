"""Offline research pipeline for CreditLens.

This module evaluates model behavior on historical/synthetic records. It does not make
real-person credit eligibility decisions. The Streamlit demo is for fictional/synthetic
applicants only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from business_analysis import approval_tradeoff, choose_cutoff, plot_tradeoff
from config import ARTIFACT_DIR, ID_COLUMN, MODEL_FEATURES, OUTPUT_DIR, PROTECTED_PROXY, RANDOM_STATE, TARGET
from data_prep import add_indian_fintech_layer, generate_demo_lending_club, prepare_dataset, run_eda
from explainability import save_shap_plots
from fairness_audit import bias_flag, fairness_metrics, fairness_score, plot_fairness
from model import choose_calibrator, evaluate_predictions, fit_model
from reject_inference import apply_legacy_policy, fuzzy_augmentation_reject_inference, legacy_cutoff


def _plot_calibration(y, raw_p, calibrated_p, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for p, label in [(raw_p, "Before"), (calibrated_p, "After")]:
        obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(pred, obs, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", label="Perfect")
    ax.set(xlabel="Predicted default probability", ylabel="Observed default rate", title="Calibration curve")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _plot_shift(before, after, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(before, bins=35, alpha=.55, label="Approved-only sample")
    ax.hist(after, bins=35, alpha=.55, label="After reject inference")
    ax.set(xlabel="Predicted default probability", ylabel="Borrowers", title="Population score shift")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _defaults(df: pd.DataFrame, features: list[str]) -> dict:
    out = {}
    for c in features:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").median()
            out[c] = float(v) if pd.notna(v) else 0.0
        else:
            mode = s.dropna().mode()
            out[c] = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
    return out


def train_creditlens(df: pd.DataFrame, output_dir=OUTPUT_DIR, artifact_dir=ARTIFACT_DIR) -> dict:
    """Train/evaluate the offline benchmark and save reproducible research artifacts."""
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True); artifact_dir.mkdir(parents=True, exist_ok=True)
    run_eda(df, output_dir)

    train_full, test = train_test_split(df, test_size=.20, stratify=df[TARGET], random_state=RANDOM_STATE)
    train, cal = train_test_split(train_full, test_size=.25, stratify=train_full[TARGET], random_state=RANDOM_STATE)
    features = MODEL_FEATURES.copy()
    ytr, ycal, yte = train[TARGET].to_numpy(), cal[TARGET].to_numpy(), test[TARGET].to_numpy()

    old_cut = legacy_cutoff(train, .70)
    a_tr, a_cal = apply_legacy_policy(train, old_cut), apply_legacy_policy(cal, old_cut)
    logit = fit_model(train.loc[a_tr], ytr[a_tr], features, "logistic")
    baseline = evaluate_predictions(yte, logit.predict_proba(test))

    ri = fuzzy_augmentation_reject_inference(train, ytr, a_tr, features)
    p_before, p_after = ri.initial_model.predict_proba(test), ri.inferred_model.predict_proba(test)
    before, after = evaluate_predictions(yte, p_before), evaluate_predictions(yte, p_after)
    _plot_shift(p_before, p_after, output_dir / "reject_inference_population_shift.png")
    pd.DataFrame([{"stage":"approved_only", **before}, {"stage":"reject_inference", **after}]).to_csv(output_dir / "reject_inference_comparison.csv", index=False)

    calibrator, cal_diag = choose_calibrator(ri.inferred_model.predict_proba(cal.loc[a_cal]), ycal[a_cal])
    raw_test = ri.inferred_model.predict_proba(test)
    p_test = calibrator.predict(raw_test)
    _plot_calibration(yte, raw_test, p_test, output_dir / "calibration_curve.png")

    fairness_cut = float(np.quantile(calibrator.predict(ri.inferred_model.predict_proba(cal)), .70))
    fair_before = fairness_metrics(yte, p_test, test[PROTECTED_PROXY], fairness_cut)
    flagged = bias_flag(fair_before)
    clean_features = [f for f in features if f not in {PROTECTED_PROXY, "pincode_income_proxy"}]
    ri2 = fuzzy_augmentation_reject_inference(train, ytr, a_tr, clean_features)
    cal2, cal_diag2 = choose_calibrator(ri2.inferred_model.predict_proba(cal.loc[a_cal]), ycal[a_cal])
    p2 = cal2.predict(ri2.inferred_model.predict_proba(test))
    fair_after = fairness_metrics(yte, p2, test[PROTECTED_PROXY], fairness_cut)
    m2 = evaluate_predictions(yte, p2)
    plot_fairness(fair_before, fair_after, output_dir / "fairness_before_after.png")
    pd.DataFrame([{"stage":"before", **fair_before}, {"stage":"proxy_removed", **fair_after}]).to_csv(output_dir / "fairness_metrics.csv", index=False)

    use_mitigation = bool(flagged and fairness_score(fair_after) < fairness_score(fair_before) and m2["roc_auc"] >= after["roc_auc"] - .02)
    if use_mitigation:
        final_model, final_cal, final_p, final_features, final_diag = ri2.inferred_model, cal2, p2, clean_features, cal_diag2
    else:
        final_model, final_cal, final_p, final_features, final_diag = ri.inferred_model, calibrator, p_test, features, cal_diag

    save_shap_plots(final_model, test[final_features], output_dir)
    curve = approval_tradeoff(yte, final_p)
    research_point = choose_cutoff(curve)
    curve.to_csv(output_dir / "approval_default_tradeoff.csv", index=False)
    plot_tradeoff(curve, float(research_point["cutoff"]), output_dir / "approval_default_tradeoff.png")

    report = {
        "data": {"rows": int(len(df)), "default_rate": float(df[TARGET].mean()), "legacy_approval_rate_train": float(a_tr.mean())},
        "baseline_logistic": baseline,
        "approved_only_xgb": before,
        "reject_inference_xgb": after,
        "calibrated_final": evaluate_predictions(yte, final_p),
        "calibration": final_diag,
        "fairness_before": fair_before,
        "fairness_after_proxy_removal": fair_after,
        "bias_flagged": flagged,
        "proxy_removal_selected_for_research_model": use_mitigation,
        "portfolio_cutoff_simulation": {
            "cutoff": float(research_point["cutoff"]),
            "approval_rate": float(research_point["approval_rate"]),
            "approved_default_rate": float(research_point["approved_default_rate"]),
            "bad_loans_avoided": int(research_point["bad_loans_avoided"]),
            "good_loans_lost": int(research_point["good_loans_lost"]),
            "loss_avoided_inr": float(research_point["loss_avoided_inr"]),
            "good_profit_lost_inr": float(research_point["good_profit_lost_inr"]),
            "net_value_inr": float(research_point["net_value_inr"]),
            "assumptions": {"avg_loan_size_inr": 150000, "lgd": .60, "good_loan_margin": .12},
            "warning": "Offline portfolio simulation only; not an applicant-level lending rule."
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({
        "model": final_model,
        "calibrator": final_cal,
        "feature_columns": final_features,
        "feature_defaults": _defaults(train, final_features),
        "metrics": report,
        "use_for": "synthetic/fictitious demonstration records only"
    }, artifact_dir / "creditlens_model.joblib")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline CreditLens research benchmark")
    parser.add_argument("--data"); parser.add_argument("--max-rows", type=int); parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        raw = generate_demo_lending_club(6000); df = add_indian_fintech_layer(raw)
        df[TARGET] = raw["loan_status"].eq("Charged Off").astype(int).to_numpy()
        df = df[[ID_COLUMN, TARGET] + MODEL_FEATURES]
    else:
        if not args.data: parser.error("--data is required unless --demo is used")
        df = prepare_dataset(args.data, args.max_rows)
    print(json.dumps(train_creditlens(df), indent=2))


if __name__ == "__main__":
    main()

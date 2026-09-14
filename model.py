"""Model training for CreditLens.

Logistic regression remains a common lending baseline because coefficient direction,
odds interpretation, stability monitoring and reason-code generation are easier to
explain to risk teams and regulators than a complex ensemble.

SMOTE is intentionally not used. Synthetic interpolation between financial records
can create implausible borrower combinations, distort class probabilities and blur
rare risk segments. Class weights change the learning objective without fabricating
new borrowers.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

TARGET = "default"
MODEL_DIR = Path("models")
OUT = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)


def split_data(df, target=TARGET, seed=42):
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor(X):
    cats = X.select_dtypes(include="object").columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, nums), ("cat", cat_pipe, cats)])


def train_models(X_train, y_train, X_val, y_val, X_test, y_test):
    prep = make_preprocessor(X_train)
    pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    baseline = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=.5))])
    baseline.fit(X_train, y_train)

    xgb_prep = make_preprocessor(X_train)
    xgb = Pipeline([("prep", xgb_prep), ("model", XGBClassifier(
        n_estimators=450, max_depth=4, learning_rate=.045, subsample=.82,
        colsample_bytree=.82, min_child_weight=8, reg_lambda=3.0,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        scale_pos_weight=pos_weight, random_state=42, n_jobs=4
    ))])
    xgb.fit(X_train, y_train)

    # Calibrate only against a held-out validation set, avoiding test leakage.
    try:
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(FrozenEstimator(xgb), method="sigmoid", cv=None)
    except ImportError:
        calibrator = CalibratedClassifierCV(xgb, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)

    metrics = {}
    for name, model in [("logistic", baseline), ("xgboost", xgb), ("calibrated_xgboost", calibrator)]:
        p = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, p)),
            "pr_auc": float(average_precision_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "accuracy_at_50pct": float(accuracy_score(y_test, p >= .5))
        }

    p_raw = xgb.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict_proba(X_test)[:, 1]
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, p_raw, n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, p_cal, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 6)); plt.plot(prob_pred_raw, prob_true_raw, marker="o", label="XGBoost")
    plt.plot(prob_pred_cal, prob_true_cal, marker="o", label="Calibrated XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed default rate"); plt.title("CreditLens calibration curve"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "calibration_curve.png", dpi=170); plt.close()

    joblib.dump(xgb, MODEL_DIR / "xgboost.joblib")
    joblib.dump(baseline, MODEL_DIR / "logistic.joblib")
    joblib.dump(calibrator, MODEL_DIR / "calibrated_xgboost.joblib")
    (MODEL_DIR / "features.json").write_text(json.dumps(X_train.columns.tolist(), indent=2))
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"xgb": xgb, "logistic": baseline, "calibrated": calibrator, "metrics": metrics}


def load_model(calibrated=True):
    name = "calibrated_xgboost.joblib" if calibrated else "xgboost.joblib"
    return joblib.load(MODEL_DIR / name)

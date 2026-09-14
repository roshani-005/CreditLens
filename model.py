"""Interpretable baseline, gradient-boosted risk model, calibration, and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import RANDOM_STATE


@dataclass
class FittedRiskModel:
    preprocessor: ColumnTransformer
    estimator: object
    feature_columns: list[str]
    model_type: str

    def transform(self, X: pd.DataFrame):
        return self.preprocessor.transform(X[self.feature_columns])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.transform(X)
        return np.asarray(self.estimator.predict_proba(Xt))[:, 1]

    def transformed_feature_names(self) -> np.ndarray:
        return self.preprocessor.get_feature_names_out()


def _split_types(X: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, features: Sequence[str], scale_numeric: bool) -> ColumnTransformer:
    numeric, categorical = _split_types(X, features)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=True)


def fit_model(
    X: pd.DataFrame,
    y: Sequence[int],
    features: Sequence[str],
    model_type: str = "xgb",
    sample_weight: Sequence[float] | None = None,
    seed: int = RANDOM_STATE,
) -> FittedRiskModel:
    """Fit using class weighting rather than SMOTE.

    SMOTE can create synthetic financial applicants with internally inconsistent
    combinations (for example impossible debt/income/account relationships), and it
    changes the empirical feature distribution used for calibration. Weighting keeps
    real rows intact and changes their loss contribution instead.
    """
    features = list(features)
    y_arr = np.asarray(y, dtype=int)
    balanced = compute_sample_weight(class_weight="balanced", y=y_arr)
    if sample_weight is not None:
        balanced = balanced * np.asarray(sample_weight, dtype=float)

    if model_type == "logistic":
        prep = build_preprocessor(X, features, scale_numeric=True)
        Xt = prep.fit_transform(X[features])
        estimator = LogisticRegression(max_iter=1200, C=0.5, random_state=seed)
        estimator.fit(Xt, y_arr, sample_weight=balanced)
    elif model_type == "xgb":
        prep = build_preprocessor(X, features, scale_numeric=False)
        Xt = prep.fit_transform(X[features])
        estimator = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=3.0,
            reg_alpha=0.2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        estimator.fit(Xt, y_arr, sample_weight=balanced)
    else:
        raise ValueError("model_type must be 'logistic' or 'xgb'")
    return FittedRiskModel(prep, estimator, features, model_type)


class ProbabilityCalibrator:
    """Small serializable probability calibrator supporting identity/Platt/isotonic."""
    def __init__(self, method: str = "identity"):
        self.method = method
        self.model = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p: Sequence[float], y: Sequence[int]):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        elif self.method != "identity":
            raise ValueError(self.method)
        return self

    def predict(self, p: Sequence[float]) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "identity" or self.model is None:
            return np.clip(p, 0, 1)
        if self.method == "platt":
            return self.model.predict_proba(self._logit(p))[:, 1]
        return np.asarray(self.model.predict(p), dtype=float)


def choose_calibrator(raw_p: Sequence[float], y: Sequence[int]) -> tuple[ProbabilityCalibrator, dict[str, float]]:
    """Select calibration only when it improves validation Brier score."""
    p = np.asarray(raw_p, dtype=float)
    y = np.asarray(y, dtype=int)
    candidates: dict[str, tuple[ProbabilityCalibrator, float]] = {
        "identity": (ProbabilityCalibrator("identity"), brier_score_loss(y, p))
    }
    for method in ("platt", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(p, y)
        candidates[method] = (cal, brier_score_loss(y, cal.predict(p)))
    # Platt scaling is the conservative default because it is strictly monotonic and
    # therefore preserves rank/AUC. Use isotonic only with a large calibration sample
    # and a material (>2% relative) Brier improvement over Platt; otherwise a tiny
    # in-sample gain is not worth the step-function ties/overfitting risk.
    raw_brier = candidates["identity"][1]
    platt_brier = candidates["platt"][1]
    iso_brier = candidates["isotonic"][1]
    if platt_brier < raw_brier - 1e-5:
        best_method = "platt"
        if len(y) >= 2000 and iso_brier < platt_brier * 0.98:
            best_method = "isotonic"
    elif len(y) >= 2000 and iso_brier < raw_brier * 0.98:
        best_method = "isotonic"
    else:
        best_method = "identity"
    diagnostics = {f"brier_{k}": float(v[1]) for k, v in candidates.items()}
    diagnostics["selected"] = best_method
    return candidates[best_method][0], diagnostics


def evaluate_predictions(y: Sequence[int], p: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted_default": float(np.mean(p)),
        "observed_default_rate": float(np.mean(y)),
    }

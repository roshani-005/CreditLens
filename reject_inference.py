"""Approved-only selection bias simulation and fuzzy-augmentation reject inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from config import RANDOM_STATE
from model import FittedRiskModel, fit_model


@dataclass
class RejectInferenceResult:
    initial_model: FittedRiskModel
    inferred_model: FittedRiskModel
    augmented_rows: int
    rejected_rows: int
    mean_inferred_bad_probability: float


def old_policy_score(df: pd.DataFrame) -> np.ndarray:
    """Simulated legacy score using only application-time variables, never the target."""
    fico = ((pd.to_numeric(df["fico_range_low"], errors="coerce") + pd.to_numeric(df["fico_range_high"], errors="coerce")) / 2).fillna(680)
    dti = pd.to_numeric(df["dti"], errors="coerce").fillna(25).clip(0, 80)
    income = pd.to_numeric(df["annual_inc"], errors="coerce").fillna(50_000).clip(10_000, 500_000)
    delinq = pd.to_numeric(df["delinq_2yrs"], errors="coerce").fillna(0).clip(0, 10)
    score = 0.58 * ((fico - 580) / 250) + 0.20 * (1 - dti / 80) + 0.15 * np.clip(np.log1p(income) / np.log(500_001), 0, 1) + 0.07 * (1 - delinq / 10)
    return np.asarray(score, dtype=float)


def legacy_cutoff(train_df: pd.DataFrame, approval_rate: float = 0.70) -> float:
    return float(np.quantile(old_policy_score(train_df), 1 - approval_rate))


def apply_legacy_policy(df: pd.DataFrame, cutoff: float) -> np.ndarray:
    return old_policy_score(df) >= cutoff


def fuzzy_augmentation_reject_inference(
    X: pd.DataFrame,
    y: Sequence[int],
    approved_mask: Sequence[bool],
    features: Sequence[str],
    rejected_weight: float = 0.55,
    seed: int = RANDOM_STATE,
) -> RejectInferenceResult:
    """Infer rejected outcomes probabilistically and retrain without pretending certainty.

    Each rejected applicant is duplicated into a good-label and bad-label parcel. The
    two copies receive weights proportional to (1-p_bad) and p_bad. This is fuzzy
    augmentation: inferred labels remain uncertain instead of becoming hard pseudo-labels.
    """
    y = np.asarray(y, dtype=int)
    approved = np.asarray(approved_mask, dtype=bool)
    if approved.sum() < 50 or (~approved).sum() < 10:
        raise ValueError("Need both a meaningful approved and rejected pool for reject inference.")

    initial = fit_model(X.loc[approved], y[approved], features, model_type="xgb", seed=seed)
    rejected_X = X.loc[~approved].copy()
    p_bad = initial.predict_proba(rejected_X)

    Xa = X.loc[approved].copy(); ya = y[approved]; wa = np.ones(len(Xa))
    Xg = rejected_X.copy(); yg = np.zeros(len(Xg), dtype=int); wg = rejected_weight * (1 - p_bad)
    Xb = rejected_X.copy(); yb = np.ones(len(Xb), dtype=int); wb = rejected_weight * p_bad

    X_aug = pd.concat([Xa, Xg, Xb], ignore_index=True)
    y_aug = np.concatenate([ya, yg, yb])
    w_aug = np.concatenate([wa, wg, wb])
    inferred = fit_model(X_aug, y_aug, features, model_type="xgb", sample_weight=w_aug, seed=seed)
    return RejectInferenceResult(
        initial_model=initial,
        inferred_model=inferred,
        augmented_rows=len(X_aug),
        rejected_rows=len(rejected_X),
        mean_inferred_bad_probability=float(np.mean(p_bad)),
    )

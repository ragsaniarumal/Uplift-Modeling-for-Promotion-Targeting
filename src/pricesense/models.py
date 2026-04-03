from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from .experiment import PROMOTION_COST, PRODUCT_REVENUE, add_treatment

FEATURES = [f"V{i}" for i in range(1, 8)]
TARGET_FRACTIONS = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_experiment(df: pd.DataFrame, seed: int = 42) -> SplitData:
    d = add_treatment(df)
    strata = d.treatment.astype(str) + "_" + d.purchase.astype(str)
    train, temp = train_test_split(d, test_size=0.40, random_state=seed, stratify=strata)
    temp_strata = temp.treatment.astype(str) + "_" + temp.purchase.astype(str)
    validation, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp_strata
    )
    return SplitData(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True))


def fit_response_model(train: pd.DataFrame) -> RandomForestClassifier:
    treated = train[train.treatment == 1]
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=100,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return model.fit(treated[FEATURES], treated.purchase)


def fit_s_learner(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=42,
    )
    return model.fit(train[FEATURES + ["treatment"]], train.purchase)


def response_score(model: RandomForestClassifier, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def uplift_score(model: HistGradientBoostingClassifier, df: pd.DataFrame) -> np.ndarray:
    x1 = df[FEATURES].copy()
    x0 = df[FEATURES].copy()
    x1["treatment"] = 1
    x0["treatment"] = 0
    return model.predict_proba(x1)[:, 1] - model.predict_proba(x0)[:, 1]


def policy_result(df: pd.DataFrame, score: np.ndarray, fraction: float) -> dict[str, float]:
    n = max(2, int(len(df) * fraction))
    order = np.argsort(-score, kind="stable")[:n]
    selected = df.iloc[order]
    treated = selected[selected.treatment == 1]
    control = selected[selected.treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Selected policy must contain treatment and control observations")
    p1 = treated.purchase.mean()
    p0 = control.purchase.mean()
    irr = p1 - p0
    expected_incremental_purchases = n * irr
    expected_nir = n * (PRODUCT_REVENUE * irr - PROMOTION_COST)
    return {
        "target_fraction": float(fraction),
        "targeted_customers": float(n),
        "promotion_purchase_rate": float(p1),
        "control_purchase_rate": float(p0),
        "incremental_response_rate": float(irr),
        "expected_incremental_purchases": float(expected_incremental_purchases),
        "expected_nir": float(expected_nir),
    }


def policy_curve(df: pd.DataFrame, score: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    for fraction in TARGET_FRACTIONS:
        row = policy_result(df, score, float(fraction))
        row["model"] = model_name
        rows.append(row)
    return pd.DataFrame(rows)


def choose_fraction(curve: pd.DataFrame) -> float:
    best = curve.sort_values(["expected_nir", "target_fraction"], ascending=[False, True]).iloc[0]
    return float(best.target_fraction)

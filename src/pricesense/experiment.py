from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

PRODUCT_REVENUE = 10.0
PROMOTION_COST = 0.15


def add_treatment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["treatment"] = (out["Promotion"] == "Yes").astype(int)
    return out


def ab_summary(df: pd.DataFrame) -> dict[str, float]:
    d = add_treatment(df)
    treated = d.loc[d.treatment == 1, "purchase"]
    control = d.loc[d.treatment == 0, "purchase"]
    p1, p0 = treated.mean(), control.mean()
    pooled = (treated.sum() + control.sum()) / len(d)
    se = np.sqrt(pooled * (1 - pooled) * (1 / len(treated) + 1 / len(control)))
    z = (p1 - p0) / se
    p_value = 2 * norm.sf(abs(z))
    irr = p1 - p0
    nir = len(d) * (PRODUCT_REVENUE * irr - PROMOTION_COST)
    return {
        "n": float(len(d)),
        "treatment_rate": float(d.treatment.mean()),
        "control_purchase_rate": float(p0),
        "promotion_purchase_rate": float(p1),
        "incremental_response_rate": float(irr),
        "z_score": float(z),
        "p_value": float(p_value),
        "blanket_expected_nir": float(nir),
    }


def bootstrap_irr_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    d = add_treatment(df)
    t = d.loc[d.treatment == 1, "purchase"].to_numpy()
    c = d.loc[d.treatment == 0, "purchase"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = rng.choice(t, len(t), replace=True).mean() - rng.choice(c, len(c), replace=True).mean()
    return tuple(np.quantile(draws, [0.025, 0.975]))

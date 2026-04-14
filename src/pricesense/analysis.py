from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import DEFAULT_PATH, download, load
from .experiment import ab_summary, bootstrap_irr_ci
from .models import (
    TARGET_FRACTIONS,
    choose_fraction,
    fit_response_model,
    fit_s_learner,
    policy_curve,
    policy_result,
    response_score,
    split_experiment,
    uplift_score,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def run(data_path: str | Path = DEFAULT_PATH) -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    path = Path(data_path)
    if not path.exists():
        download(path)
    df = load(path)

    overall = ab_summary(df)
    ci_low, ci_high = bootstrap_irr_ci(df, n_boot=1000, seed=42)
    overall["irr_ci_low"] = ci_low
    overall["irr_ci_high"] = ci_high

    split = split_experiment(df)
    response = fit_response_model(split.train)
    uplift = fit_s_learner(split.train)

    val_response = policy_curve(split.validation, response_score(response, split.validation), "response_model")
    val_uplift = policy_curve(split.validation, uplift_score(uplift, split.validation), "uplift_s_learner")
    val_curve = pd.concat([val_response, val_uplift], ignore_index=True)
    val_curve.to_csv(ARTIFACTS / "validation_policy_curve.csv", index=False, float_format="%.8f")

    response_fraction = choose_fraction(val_response)
    uplift_fraction = choose_fraction(val_uplift)

    test_response_score = response_score(response, split.test)
    test_uplift_score = uplift_score(uplift, split.test)
    test_response = policy_curve(split.test, test_response_score, "response_model")
    test_uplift = policy_curve(split.test, test_uplift_score, "uplift_s_learner")
    test_curve = pd.concat([test_response, test_uplift], ignore_index=True)
    test_curve.to_csv(ARTIFACTS / "test_policy_curve.csv", index=False, float_format="%.8f")

    response_holdout = policy_result(split.test, test_response_score, response_fraction)
    uplift_holdout = policy_result(split.test, test_uplift_score, uplift_fraction)
    response_holdout["selected_from_validation"] = response_fraction
    uplift_holdout["selected_from_validation"] = uplift_fraction

    summary = {
        "experiment": overall,
        "split_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "response_model": response_holdout,
        "uplift_s_learner": uplift_holdout,
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Segment-level observed lift using quantiles of uplift score on the holdout.
    scored = split.test[["treatment", "purchase"]].copy()
    scored["uplift_score"] = test_uplift_score
    scored["uplift_band"] = pd.qcut(scored.uplift_score.rank(method="first"), 5, labels=False) + 1
    seg_rows = []
    for band, g in scored.groupby("uplift_band"):
        p1 = g.loc[g.treatment == 1, "purchase"].mean()
        p0 = g.loc[g.treatment == 0, "purchase"].mean()
        seg_rows.append({"uplift_band": int(band), "observed_irr": p1 - p0, "n": len(g)})
    segments = pd.DataFrame(seg_rows)
    segments.to_csv(ARTIFACTS / "uplift_segments.csv", index=False, float_format="%.8f")

    # Static PNG for GitHub README.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, curve in [("Response model", test_response), ("Uplift S-learner", test_uplift)]:
        ax.plot(curve.target_fraction * 100, curve.expected_nir, marker="o", label=name)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Customers targeted (%)")
    ax.set_ylabel("Expected net incremental revenue ($)")
    ax.set_title("Holdout promotion policy by targeting budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "policy_curve.png", dpi=160)
    plt.close(fig)

    # Standalone interactive report; no web server required.
    dash = make_subplots(rows=1, cols=2, subplot_titles=("Policy value", "Observed uplift by score band"))
    for name, curve in [("Response", test_response), ("Uplift", test_uplift)]:
        dash.add_trace(
            go.Scatter(x=curve.target_fraction * 100, y=curve.expected_nir, mode="lines+markers", name=name),
            row=1, col=1,
        )
    dash.add_trace(
        go.Bar(x=[f"Q{b}" for b in segments.uplift_band], y=segments.observed_irr * 100, name="Observed IRR (pp)"),
        row=1, col=2,
    )
    dash.update_xaxes(title_text="Customers targeted (%)", row=1, col=1)
    dash.update_yaxes(title_text="Expected NIR ($)", row=1, col=1)
    dash.update_yaxes(title_text="Incremental response (percentage points)", row=1, col=2)
    dash.update_layout(title="PriceSense — promotion targeting explorer", template="plotly_white")
    dash.write_html(REPORTS / "dashboard.html", include_plotlyjs=True, div_id="pricesense-dashboard")

    return summary


def main() -> None:
    summary = run()
    exp = summary["experiment"]
    print(f"Promotion purchase rate: {exp['promotion_purchase_rate']:.3%}")
    print(f"Control purchase rate: {exp['control_purchase_rate']:.3%}")
    print(f"Incremental response: {exp['incremental_response_rate']:.3%}")
    print(f"Blanket expected NIR: ${exp['blanket_expected_nir']:.2f}")
    print(f"Response policy holdout NIR: ${summary['response_model']['expected_nir']:.2f}")
    print(f"Uplift policy holdout NIR: ${summary['uplift_s_learner']['expected_nir']:.2f}")


if __name__ == "__main__":
    main()

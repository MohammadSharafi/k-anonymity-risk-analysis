from __future__ import annotations

import numpy as np
import pandas as pd


def compute_utility_metrics(original_df: pd.DataFrame, release_df: pd.DataFrame) -> dict[str, float]:
    original_rows = len(original_df)
    retained_rows = len(release_df)

    if retained_rows == 0:
        return {
            "retained_rows": 0,
            "retention_rate": 0.0,
            "suppression_rate": 1.0,
            "income_distribution_shift": 1.0,
            "hours_mean_abs_diff": float(original_df["hours_per_week"].mean()),
            "hours_variance_abs_diff": float(original_df["hours_per_week"].var()),
            "income_gap_by_sex": 1.0,
        }

    original_income_rate = float(original_df["income_gt_50k"].mean())
    release_income_rate = float(release_df["income_gt_50k"].mean())

    utility = {
        "retained_rows": retained_rows,
        "retention_rate": retained_rows / original_rows if original_rows else 0.0,
        "suppression_rate": 1 - (retained_rows / original_rows if original_rows else 0.0),
        "income_distribution_shift": abs(original_income_rate - release_income_rate),
        "hours_mean_abs_diff": abs(
            float(original_df["hours_per_week"].mean()) - float(release_df["hours_per_week"].mean())
        ),
        "hours_variance_abs_diff": abs(
            _safe_variance(original_df["hours_per_week"]) - _safe_variance(release_df["hours_per_week"])
        ),
        "income_gap_by_sex": _income_gap_by_sex(original_df, release_df),
    }
    return utility


def _safe_variance(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    return float(series.var())


def _income_gap_by_sex(original_df: pd.DataFrame, release_df: pd.DataFrame) -> float:
    comparable = release_df[release_df["sex"].isin(["Male", "Female"])].copy()
    if comparable.empty:
        return 1.0

    original_rates = original_df.groupby("sex", observed=False)["income_gt_50k"].mean()
    release_rates = comparable.groupby("sex", observed=False)["income_gt_50k"].mean()

    aligned = pd.concat([original_rates, release_rates], axis=1, keys=["original", "release"]).fillna(0.0)
    return float(np.abs(aligned["original"] - aligned["release"]).mean())

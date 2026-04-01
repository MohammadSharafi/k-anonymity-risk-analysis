from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .generalization import generalize_frame, max_level


@dataclass
class AnonymizationResult:
    release_df: pd.DataFrame
    generalization_levels: dict[str, int]
    strategy: str
    k: int
    qi_columns: list[str]
    rows_removed: int

    @property
    def suppression_rate(self) -> float:
        if self.release_df.attrs.get("original_row_count", 0) == 0:
            return 0.0
        return self.rows_removed / self.release_df.attrs["original_row_count"]


def anonymize_dataset(
    df: pd.DataFrame,
    qi_columns: list[str],
    k: int,
    strategy: str,
) -> AnonymizationResult:
    if strategy not in {"generalization_first", "targeted_suppression"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    original_row_count = len(df)
    levels = {column: 0 for column in qi_columns}
    order = _generalization_order(qi_columns)

    if strategy == "generalization_first":
        release_df, levels = _run_generalization_first(df, qi_columns, k, levels, order)
    else:
        release_df, levels = _run_targeted_suppression(df, qi_columns, k, levels, order)

    release_df = release_df.copy()
    release_df.attrs["original_row_count"] = original_row_count

    return AnonymizationResult(
        release_df=release_df,
        generalization_levels=levels,
        strategy=strategy,
        k=k,
        qi_columns=qi_columns,
        rows_removed=original_row_count - len(release_df),
    )


def _run_generalization_first(
    df: pd.DataFrame,
    qi_columns: list[str],
    k: int,
    levels: dict[str, int],
    order: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    while True:
        generalized = generalize_frame(df, qi_columns, levels)
        violating = _violating_rows_mask(generalized, qi_columns, k)
        if not violating.any():
            return generalized, levels

        next_levels = _best_generalization_step(df, qi_columns, k, levels, order)
        if next_levels is None:
            return _suppress_small_classes(generalized, qi_columns, k), levels
        levels = next_levels


def _run_targeted_suppression(
    df: pd.DataFrame,
    qi_columns: list[str],
    k: int,
    levels: dict[str, int],
    order: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stage_limits = {column: min(1, max_level(column)) for column in qi_columns}

    while True:
        generalized = generalize_frame(df, qi_columns, levels)
        release_df = _suppress_small_classes(generalized, qi_columns, k)

        if not release_df.empty:
            return release_df, levels

        upgraded = False
        for column in order:
            if levels[column] < stage_limits[column]:
                levels[column] += 1
                upgraded = True
                break

        if upgraded:
            continue

        expanded = False
        for column in order:
            if stage_limits[column] < max_level(column):
                stage_limits[column] += 1
                expanded = True
                break

        if not expanded:
            return release_df, levels


def _violating_rows_mask(df: pd.DataFrame, qi_columns: list[str], k: int) -> pd.Series:
    sizes = df.groupby(qi_columns, dropna=False, observed=False)[qi_columns[0]].transform("size")
    return sizes < k


def _suppress_small_classes(df: pd.DataFrame, qi_columns: list[str], k: int) -> pd.DataFrame:
    violating = _violating_rows_mask(df, qi_columns, k)
    return df.loc[~violating].copy()


def _generalization_order(qi_columns: list[str]) -> list[str]:
    preferred = ["age", "education", "marital_status", "occupation", "native_country", "sex"]
    ordered = [column for column in preferred if column in qi_columns]
    ordered.extend(column for column in qi_columns if column not in ordered)
    return ordered


def _best_generalization_step(
    df: pd.DataFrame,
    qi_columns: list[str],
    k: int,
    levels: dict[str, int],
    order: list[str],
) -> dict[str, int] | None:
    candidates: list[tuple[int, float, int, dict[str, int]]] = []

    for rank, column in enumerate(order):
        if levels[column] >= max_level(column):
            continue

        trial_levels = levels.copy()
        trial_levels[column] += 1
        generalized = generalize_frame(df, qi_columns, trial_levels)
        violating_rows = int(_violating_rows_mask(generalized, qi_columns, k).sum())
        information_loss = _information_loss_score(trial_levels, qi_columns)
        candidates.append((violating_rows, information_loss, rank, trial_levels))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _information_loss_score(levels: dict[str, int], qi_columns: list[str]) -> float:
    score = 0.0
    for column in qi_columns:
        maximum = max_level(column)
        if maximum == 0:
            continue
        score += levels[column] / maximum
    return score

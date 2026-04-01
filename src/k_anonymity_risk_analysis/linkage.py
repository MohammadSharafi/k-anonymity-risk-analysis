from __future__ import annotations

import pandas as pd

from .config import RANDOM_STATE
from .generalization import generalize_frame


def evaluate_linkage_risk(
    original_df: pd.DataFrame,
    release_df: pd.DataFrame,
    qi_columns: list[str],
    levels: dict[str, int],
    sample_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    auxiliary_df = create_auxiliary_dataset(
        original_df,
        qi_columns=qi_columns,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )

    transformed_aux = generalize_frame(auxiliary_df, qi_columns, levels)

    if release_df.empty:
        no_matches = len(transformed_aux)
        return {
            "auxiliary_size": len(transformed_aux),
            "unique_matches": 0,
            "ambiguous_matches": 0,
            "no_matches": no_matches,
            "unique_match_rate": 0.0,
            "ambiguous_match_rate": 0.0,
            "no_match_rate": 1.0,
        }

    release_key_counts = (
        release_df.groupby(qi_columns, dropna=False, observed=False)
        .size()
        .rename("match_count")
        .reset_index()
    )

    linked = transformed_aux.merge(release_key_counts, on=qi_columns, how="left")
    linked["match_count"] = linked["match_count"].fillna(0).astype(int)

    unique_matches = int((linked["match_count"] == 1).sum())
    ambiguous_matches = int((linked["match_count"] > 1).sum())
    no_matches = int((linked["match_count"] == 0).sum())
    total = len(linked)

    return {
        "auxiliary_size": total,
        "unique_matches": unique_matches,
        "ambiguous_matches": ambiguous_matches,
        "no_matches": no_matches,
        "unique_match_rate": unique_matches / total if total else 0.0,
        "ambiguous_match_rate": ambiguous_matches / total if total else 0.0,
        "no_match_rate": no_matches / total if total else 0.0,
    }


def create_auxiliary_dataset(
    df: pd.DataFrame,
    qi_columns: list[str],
    sample_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    sample = df.sample(frac=sample_fraction, random_state=random_state).copy()
    return sample[["person_id", *qi_columns]]

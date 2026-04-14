from __future__ import annotations

import pandas as pd


def equivalence_class_sizes(df: pd.DataFrame, qi_columns: list[str]) -> pd.Series:
    """Return equivalence-class sizes for the given quasi-identifiers."""
    if df.empty:
        return pd.Series(dtype="int64")
    return df.groupby(qi_columns, dropna=False, observed=False).size()


def validate_k_anonymity(df: pd.DataFrame, qi_columns: list[str], k: int) -> dict[str, float | bool]:
    """Check whether every visible QI combination appears at least k times."""
    class_sizes = equivalence_class_sizes(df, qi_columns)
    if class_sizes.empty:
        return {
            "is_k_anonymous": True,
            "min_equivalence_class_size": 0,
            "num_equivalence_classes": 0,
            "num_violating_classes": 0,
        }

    min_size = int(class_sizes.min())
    violating = int((class_sizes < k).sum())
    return {
        "is_k_anonymous": violating == 0,
        "min_equivalence_class_size": min_size,
        "num_equivalence_classes": int(class_sizes.shape[0]),
        "num_violating_classes": violating,
    }

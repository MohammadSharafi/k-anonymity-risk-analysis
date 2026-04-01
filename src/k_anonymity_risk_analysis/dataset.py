from pathlib import Path

import pandas as pd

from .config import ADULT_COLUMNS, CATEGORICAL_COLUMNS, DEFAULT_DATA_PATH


def load_adult_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and lightly clean the Adult Census Income dataset."""
    csv_path = Path(path)
    df = pd.read_csv(
        csv_path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

    for column in ["workclass", "occupation", "native_country"]:
        df[column] = df[column].fillna("Unknown")

    df["person_id"] = [f"P{i:05d}" for i in range(1, len(df) + 1)]
    df["income_gt_50k"] = (df["income"] == ">50K").astype(int)
    return df

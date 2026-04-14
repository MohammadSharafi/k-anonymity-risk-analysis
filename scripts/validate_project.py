from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CACHE_DIR = PROJECT_ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import pandas as pd

from k_anonymity_risk_analysis import K_VALUES, QI_SETS, STRATEGIES, validate_k_anonymity


def main() -> None:
    releases_dir = PROJECT_ROOT / "results" / "releases"
    summary_path = PROJECT_ROOT / "results" / "metrics" / "experiment_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    expected_rows = len(QI_SETS) * len(STRATEGIES) * len(K_VALUES)
    if len(summary_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} summary rows, found {len(summary_df)}")

    failures: list[str] = []
    validations: list[str] = []

    for _, row in summary_df.iterrows():
        qi_set = row["qi_set"]
        strategy = row["strategy"]
        k = int(row["k"])
        release_path = releases_dir / f"{qi_set}_{strategy}_k{k}.csv"
        if not release_path.exists():
            failures.append(f"Missing release file: {release_path}")
            continue

        release_df = pd.read_csv(release_path)
        validation = validate_k_anonymity(release_df, QI_SETS[qi_set], k)
        validations.append(
            f"{qi_set} | {strategy} | k={k} | min_class={validation['min_equivalence_class_size']} | "
            f"violating={validation['num_violating_classes']}"
        )
        if not validation["is_k_anonymous"]:
            failures.append(f"k-anonymity violation in {release_path}")

    print("Validation summary:")
    for line in validations:
        print(line)

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print("\nAll releases passed validation.")


if __name__ == "__main__":
    main()

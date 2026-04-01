from __future__ import annotations

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from k_anonymity_risk_analysis import (
    DEFAULT_DATA_PATH,
    K_VALUES,
    QI_SETS,
    STRATEGIES,
    anonymize_dataset,
    compute_utility_metrics,
    evaluate_linkage_risk,
    load_adult_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-anonymity experiments for the Adult dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_adult_dataset(args.data_path)

    figures_dir = args.results_dir / "figures"
    metrics_dir = args.results_dir / "metrics"
    releases_dir = args.results_dir / "releases"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    releases_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for qi_set_name, qi_columns in QI_SETS.items():
        for strategy in STRATEGIES:
            for k in K_VALUES:
                result = anonymize_dataset(df, qi_columns=qi_columns, k=k, strategy=strategy)
                linkage_metrics = evaluate_linkage_risk(
                    original_df=df,
                    release_df=result.release_df,
                    qi_columns=qi_columns,
                    levels=result.generalization_levels,
                    sample_fraction=args.sample_fraction,
                )
                utility_metrics = compute_utility_metrics(df, result.release_df)

                release_path = releases_dir / f"{qi_set_name}_{strategy}_k{k}.csv"
                result.release_df.to_csv(release_path, index=False)

                summary_rows.append(
                    {
                        "qi_set": qi_set_name,
                        "strategy": strategy,
                        "k": k,
                        "qi_columns": ",".join(qi_columns),
                        "rows_removed": result.rows_removed,
                        "generalization_levels": _format_levels(result.generalization_levels),
                        **linkage_metrics,
                        **utility_metrics,
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["qi_set", "strategy", "k"]).reset_index(drop=True)
    summary_df.to_csv(metrics_dir / "experiment_summary.csv", index=False)

    _plot_metric(
        summary_df=summary_df,
        metric="unique_match_rate",
        ylabel="Unique Match Rate",
        title="Re-identification Risk vs k",
        output_path=figures_dir / "risk_vs_k.png",
    )
    _plot_metric(
        summary_df=summary_df,
        metric="income_distribution_shift",
        ylabel="Income Distribution Shift",
        title="Utility Loss vs k",
        output_path=figures_dir / "utility_vs_k.png",
    )

    print(f"Saved summary to {metrics_dir / 'experiment_summary.csv'}")
    print(f"Saved figures to {figures_dir}")
    print(f"Saved anonymized releases to {releases_dir}")


def _plot_metric(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    for (qi_set, strategy), group in summary_df.groupby(["qi_set", "strategy"], observed=False):
        ordered = group.sort_values("k")
        plt.plot(
            ordered["k"],
            ordered[metric],
            marker="o",
            label=f"{qi_set} | {strategy}",
        )

    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(summary_df["k"].unique()))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _format_levels(levels: dict[str, int]) -> str:
    return ",".join(f"{column}:{level}" for column, level in levels.items())


if __name__ == "__main__":
    main()

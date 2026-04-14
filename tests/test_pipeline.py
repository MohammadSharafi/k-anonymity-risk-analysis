from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from k_anonymity_risk_analysis import (
    K_VALUES,
    QI_SETS,
    STRATEGIES,
    anonymize_dataset,
    compute_utility_metrics,
    evaluate_linkage_risk,
    load_adult_dataset,
    validate_k_anonymity,
)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_adult_dataset().head(5000).copy()

    def test_dataset_has_expected_helper_columns(self) -> None:
        self.assertIn("person_id", self.df.columns)
        self.assertIn("income_gt_50k", self.df.columns)

    def test_all_runs_satisfy_k_anonymity(self) -> None:
        for qi_columns in QI_SETS.values():
            for strategy in STRATEGIES:
                for k in K_VALUES:
                    with self.subTest(qi_columns=qi_columns, strategy=strategy, k=k):
                        result = anonymize_dataset(self.df, qi_columns=qi_columns, k=k, strategy=strategy)
                        validation = validate_k_anonymity(result.release_df, qi_columns, k)
                        self.assertTrue(validation["is_k_anonymous"])

    def test_linkage_unique_match_rate_is_zero_after_anonymization(self) -> None:
        result = anonymize_dataset(self.df, qi_columns=QI_SETS["set_a"], k=5, strategy="targeted_suppression")
        metrics = evaluate_linkage_risk(
            original_df=self.df,
            release_df=result.release_df,
            qi_columns=QI_SETS["set_a"],
            levels=result.generalization_levels,
            sample_fraction=0.2,
            random_state=42,
        )
        self.assertEqual(metrics["unique_match_rate"], 0.0)

    def test_utility_metrics_include_information_loss(self) -> None:
        result = anonymize_dataset(self.df, qi_columns=QI_SETS["set_b"], k=10, strategy="generalization_first")
        metrics = compute_utility_metrics(
            self.df,
            result.release_df,
            qi_columns=QI_SETS["set_b"],
            generalization_levels=result.generalization_levels,
        )
        self.assertIn("information_loss_score", metrics)
        self.assertGreaterEqual(metrics["information_loss_score"], 0.0)
        self.assertLessEqual(metrics["information_loss_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

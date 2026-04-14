"""Utilities for the MSCS 714 k-anonymity project."""

from .anonymizer import AnonymizationResult, anonymize_dataset
from .config import DEFAULT_DATA_PATH, K_VALUES, QI_SETS, STRATEGIES
from .dataset import load_adult_dataset
from .linkage import evaluate_linkage_risk
from .utility import compute_utility_metrics
from .validation import validate_k_anonymity

__all__ = [
    "AnonymizationResult",
    "DEFAULT_DATA_PATH",
    "K_VALUES",
    "QI_SETS",
    "STRATEGIES",
    "anonymize_dataset",
    "compute_utility_metrics",
    "evaluate_linkage_risk",
    "load_adult_dataset",
    "validate_k_anonymity",
]

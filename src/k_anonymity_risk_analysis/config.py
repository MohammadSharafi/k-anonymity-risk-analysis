from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult" / "adult.data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

QI_SETS = {
    "set_a": ["age", "sex", "education", "marital_status"],
    "set_b": ["age", "sex", "occupation", "native_country"],
}

K_VALUES = [2, 5, 10]
STRATEGIES = ["generalization_first", "targeted_suppression"]
SENSITIVE_ATTRIBUTE = "income"
RANDOM_STATE = 42

"""Central configuration: paths, random seed, split settings and model list."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
FEATURE_IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"

RAW_DATA_FILE = RAW_DATA_DIR / "metadata.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
TARGET_COLUMN = "figo_label"
PATIENT_ID_COLUMN = "patient_id"  # used for patient-level splitting
RECORD_ID_COLUMN = "record_id"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Patient-level split: 70% train, 15% validation, 15% test
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Feature selection: top-k values to compare (plus the full 103 features)
TOP_K_VALUES = [10, 20, 30, 50]
CORRELATION_THRESHOLD = 0.90

# Model selection: F1-score of the abnormal class (figo_label = 1)
SCORING = "f1"

MODELS = [
    "logistic_regression",
    "k_nearest_neighbors",
    "support_vector_machine",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
]

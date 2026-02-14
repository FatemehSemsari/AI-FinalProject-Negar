from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_CSV = PROCESSED_DIR / "cybersecurity_intrusion_train_preprocessed.csv"
TEST_CSV  = PROCESSED_DIR / "cybersecurity_intrusion_test_preprocessed.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CHARTS_DIR  = RESULTS_DIR / "charts"
MODELS_DIR = PROJECT_ROOT / "trained_model"
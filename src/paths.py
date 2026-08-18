from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA = RAW_DATA_DIR / "online_retail_raw.csv"
FEATURES_TARGETS = PROCESSED_DATA_DIR / "customer_features_targets.csv"
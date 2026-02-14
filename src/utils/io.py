import json
from pathlib import Path
import pandas as pd
import joblib
from pathlib import Path

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def save_model(model, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(model, path)

def load_model(path: Path):
    return joblib.load(path)
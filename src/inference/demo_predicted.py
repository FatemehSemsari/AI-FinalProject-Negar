import pandas as pd
import numpy as np

from src.utils.paths import MODELS_DIR
from src.utils.io import load_model
from src.training.feature_engineering import add_security_featchers

MODEL_NAME = "xgb_v1"
THRESHOLD = 0.35

def main():
    model = load_model(MODELS_DIR / f"{MODEL_NAME}.joblib")

    df = pd.read_csv("data/demo/demo_input_preprocessed.csv")
    # df = df.drop(columns=["attack_detected"])

    df = add_security_featchers(df)

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    out = df.copy()
    out["attack_probability"] = proba
    out["attack_predicted"] = pred

    print(out.head(10))

if __name__ == "__main__":
    main()
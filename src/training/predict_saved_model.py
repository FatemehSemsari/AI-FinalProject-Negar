import numpy as np
from src.utils.paths import MODELS_DIR, TEST_CSV, TRAIN_CSV
from src.training.train_xgb import load_preprocessed_train_test
from src.training.feature_engineering import add_security_featchers
from src.utils.io import load_model

TARGET_COL = "attack_detected"
MODEL_NAME = "xgb_v1"

def main():
    X, y, X_test, y_test = load_preprocessed_train_test(TRAIN_CSV, TEST_CSV)
    X_test = add_security_featchers(X_test)
    model = load_model(MODELS_DIR / f"{MODEL_NAME}.joblib")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("Loaded model OK. preds:", np.bincount(y_pred))

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.training.train_xgb import load_preprocessed_train_test
from src.training.feature_engineering import add_security_featchers
from src.utils.seed import set_seed
from src.utils.io import save_json, ensure_dir
from src.evaluation.metrics import compute_classification_metrics
from src.models.xgb_model import xgb_v1_fe_tuned
from src.evaluation.plots import plot_confusion_matrix, plot_roc, plot_training_logloss
from src.utils.paths import TRAIN_CSV, TEST_CSV, METRICS_DIR, CHARTS_DIR

TARGET_COL = "attack_detected"
RANDOM_STATE = 42
EXPERIMENT_NAME = "xgb_v1"
VAL_SIZE = 0.2

def main():
    set_seed(RANDOM_STATE)

    metrics_path = METRICS_DIR / "phase2" / f"{EXPERIMENT_NAME}_metrics.json"
    charts_dir = CHARTS_DIR / "phase2" / EXPERIMENT_NAME
    ensure_dir(metrics_path.parent)
    ensure_dir(charts_dir)

    X, y, X_test, y_test = load_preprocessed_train_test(TRAIN_CSV, TEST_CSV)

    X = add_security_featchers(X)
    X_test=add_security_featchers(X_test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


    print("Shapes:")
    print("  Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Train class distribution:")
    print(y_train.value_counts())

    model = xgb_v1_fe_tuned()

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

   

    # Validation evaluation
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_classification_metrics(y_val, y_pred_val, y_proba_val)
    plot_confusion_matrix(
        y_val, y_pred_val,
        out_path=charts_dir / "confusion_matrix_val.png",
        title=f"Confusion Matrix (VAL) - {EXPERIMENT_NAME}"
    )
    plot_roc(
        y_val, y_proba_val,
        out_path=charts_dir / "roc_curve_val.png",
        title=f"ROC Curve (VAL) - {EXPERIMENT_NAME}"
    )

    # Test evaluation 
    # decresing threshold
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test > 0.35).astype(int)
  
    test_metrics = compute_classification_metrics(y_test, y_pred_test, y_proba_test)
    plot_confusion_matrix(
        y_test, y_pred_test,
        out_path=charts_dir / "confusion_matrix_test.png",
        title=f"Confusion Matrix (TEST) - {EXPERIMENT_NAME}"
    )
    plot_roc(
        y_test, y_proba_test,
        out_path=charts_dir / "roc_curve_test.png",
        title=f"ROC Curve (TEST) - {EXPERIMENT_NAME}"
    )

    # Training curve (logloss)
    evals_result = model.evals_result()
    plot_training_logloss(
        evals_result,
        out_path=charts_dir / "training_logloss_curve.png",
        title=f"Training Curve (logloss) - {EXPERIMENT_NAME}"
    )

    payload = {
        "model": "XGBClassifier_v1",
        "experiment": EXPERIMENT_NAME,
        "random_state": RANDOM_STATE,
        "data": {
            "train_csv": str(TRAIN_CSV),
            "test_csv": str(TEST_CSV),
            "val_size": VAL_SIZE
        },
        "params": model.get_params(),
        "best_iteration": None,
        "validation": val_metrics,
        "test": test_metrics
    }

    save_json(metrics_path, payload)

    print("\nSaved metrics:", metrics_path)
    print("Saved charts to:", charts_dir)
    print("\nValidation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
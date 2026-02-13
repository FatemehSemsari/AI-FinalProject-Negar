import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.utils.paths import TRAIN_CSV, TEST_CSV
from sklearn.model_selection import GridSearchCV
from src.utils.io import save_json, ensure_dir
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc, plot_training_logloss

def logistic_model():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    X_train = df_train.drop("attack_detected", axis=1)
    y_train = df_train["attack_detected"]
    X_test = df_test.drop("attack_detected", axis=1)
    y_test = df_test["attack_detected"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid= {"C": [0.01, 0.1, 1,10,100]}, cv=5, scoring="f1")
    grid.fit(X_train, y_train)
    best_reg_model = grid.best_estimator_

    y_pred = best_reg_model.predict(X_test)
    y_proba = best_reg_model.predict_proba(X_test)[:,1]

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    results = {
       "model": "LogisticRegression",
        "split": "test",
        "params": grid.best_params_,
        **metrics
    }
    save_json(Path("results/metrics/phase2/logistic_metrics.json"),results)
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, out_path=Path("results/charts/phase2/logistic/confusion_matrix.png"),title="Logistic Regression - Confusion Matrix")
    plot_roc(y_true=y_test, y_proba=y_proba, out_path=Path("results/charts/phase2/logistic/roc_curve.png"), title="Logistic Regression - ROC Curve")

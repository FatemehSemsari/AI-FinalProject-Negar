import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.logistic_model import logistic_model


def compare_models():
    files = {
       "XGBoost": "results/metrics/phase2/xgb_v0_metrics.json",
    "RandomForest":"results/metrics/phase-1/baseline_metrics.json",
    "LogisticRegression": "results/metrics/phase2/logistic_metrics.json"
    }
    results = []
    for model_name, path in files.items():
        with open(path) as f:
            data = json.load(f)
        
        if "test" in data:
            test_data = data["test"]
        else:
            test_data = data

        results.append({
            "Model": model_name,
            "Accuracy": test_data["accuracy"],
            "Precision": test_data["precision"],
            "Recall": test_data["recall"],
            "F1": test_data["f1"],
            "ROC_AUC": test_data["roc_auc"]
        })
    df = pd.DataFrame(results)
    df.set_index("Model", inplace=True)
    output_path = Path("results/charts/phase2")
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

    for metric in metrics:
        plt.figure(figsize=(6,4))
        plt.bar(df.index, df[metric])
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison Between Models")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric.lower()}_comparison.png")
        plt.close()


def main():
    logistic_model()
    compare_models()

 
if __name__ == "__main__":
    main()
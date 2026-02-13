import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from src.utils.io import ensure_dir

def plot_confusion_matrix(y_true, y_pred, out_path: Path, title: str = "") -> None:
    ensure_dir(out_path.parent)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Attack"])
    plt.yticks(tick_marks, ["Normal", "Attack"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, y_proba, out_path: Path, title: str = "") -> float:
    ensure_dir(out_path.parent)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return float(auc)


def plot_training_logloss(evals_result: dict, out_path: Path, title: str = "") -> None:
    ensure_dir(out_path.parent)
    train_loss = evals_result.get("validation_0", {}).get("logloss", [])
    val_loss   = evals_result.get("validation_1", {}).get("logloss", [])
    if not train_loss or not val_loss:
        return
    plt.figure(figsize=(6, 5))
    plt.plot(train_loss, label="train logloss")
    plt.plot(val_loss, label="val logloss")
    plt.xlabel("Iteration")
    plt.ylabel("Logloss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
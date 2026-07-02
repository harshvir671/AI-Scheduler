

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "num_processes",
    "avg_arrival_gap",
    "avg_burst",
    "burst_variance",
    "min_burst",
    "max_burst",
    "avg_priority",
    "priority_variance",
    "avg_memory",
    "cpu_load_estimate",
]
TARGET_COLUMN = "best_algorithm"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "training_metrics.json")
CM_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "confusion_matrix.png")
FI_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "feature_importance.png")


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def train(csv_path: str, test_size: float = 0.2, random_state: int = 42, n_estimators: int = 300):
    df = load_data(csv_path)

    X = df[FEATURE_COLUMNS].values
    y_raw = df[TARGET_COLUMN].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",   # dataset is imbalanced (SJF wins most often)
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)

    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print(f"Test accuracy:       {acc:.4f}")
    print(f"Macro F1:            {f1_macro:.4f}")
    print(f"Weighted F1:         {f1_weighted:.4f}")
    print(f"5-fold CV accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Feature importances
    importances = sorted(
        zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda x: -x[1]
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:20s} {imp:.4f}")

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved label encoder to {ENCODER_PATH}")

    metrics_out = {
        "test_accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "classes": list(encoder.classes_),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": {name: float(imp) for name, imp in importances},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_estimators": n_estimators,
        "class_distribution": pd.Series(y_raw).value_counts().to_dict(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    # Plots (best-effort; skip gracefully if matplotlib unavailable)
    try:
        _save_plots(cm, encoder.classes_, importances)
    except Exception as e:
        print(f"(skipped plot generation: {e})")

    return clf, encoder, metrics_out


def _save_plots(cm, class_names, importances):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=130)
    plt.close(fig)

    # Feature importance bar chart
    names = [n for n, _ in importances]
    vals = [v for _, v in importances]
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(names[::-1], vals[::-1], color="#4f7cff")
    ax2.set_xlabel("Importance")
    ax2.set_title("Feature Importance (Random Forest)")
    fig2.tight_layout()
    fig2.savefig(FI_PATH, dpi=130)
    plt.close(fig2)

    print(f"Saved confusion matrix plot to {CM_PATH}")
    print(f"Saved feature importance plot to {FI_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "..", "data", "workloads.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    train(args.csv, test_size=args.test_size, n_estimators=args.n_estimators)

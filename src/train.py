"""
Train and compare SVM, Decision Tree, and Random Forest models on the
diabetes prediction dataset, then save the best pipeline (preprocessing
+ model bundled together) to models/pipeline.pkl.

Usage:
    python src/train.py
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from pipeline import ALL_FEATURES, TARGET, build_model_pipeline

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes_prediction_dataset.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["not diabetic", "diabetic"]))

    return metrics, confusion_matrix(y_test, y_pred).tolist()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Positive rate (train): {y_train.mean():.3%}")

    # class_weight='balanced' matters here: only ~8.5% of rows are positive.
    candidates = {
        "SVM": svm.SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Kernel SVM training cost grows roughly quadratically with sample count.
    # On 80k rows a full-data RBF fit is impractically slow for this comparison
    # step, so SVM trains on a stratified subsample. Decision Tree and Random
    # Forest still train on the full training set.
    SVM_SAMPLE_SIZE = 8000
    X_train_svm, _, y_train_svm, _ = train_test_split(
        X_train,
        y_train,
        train_size=SVM_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    results = {}
    fitted_pipelines = {}

    for name, clf in candidates.items():
        pipe = build_model_pipeline(clf)
        start = time.time()
        if name == "SVM":
            pipe.fit(X_train_svm, y_train_svm)
        else:
            pipe.fit(X_train, y_train)
        elapsed = time.time() - start
        metrics, cm = evaluate(name, pipe, X_test, y_test)
        metrics["train_seconds"] = round(elapsed, 2)
        results[name] = {"metrics": metrics, "confusion_matrix": cm}
        fitted_pipelines[name] = pipe

    # Pick best model by F1 (accuracy alone is misleading on imbalanced data)
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1"])
    best_pipe = fitted_pipelines[best_name]

    print(f"\nBest model by F1 score: {best_name}")

    joblib.dump(best_pipe, MODELS_DIR / "pipeline.pkl")
    with open(MODELS_DIR / "model_info.json", "w") as f:
        json.dump({"best_model": best_name, "features": ALL_FEATURES}, f, indent=2)
    with open(METRICS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved pipeline to {MODELS_DIR / 'pipeline.pkl'}")
    print(f"Saved metrics to {METRICS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()

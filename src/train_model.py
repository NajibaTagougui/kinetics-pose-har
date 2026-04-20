"""
train_model.py
--------------
Train a Random Forest classifier on the 132-dimensional pose feature vectors.

Results reported in the paper
    4-class benchmark: 76.55% accuracy / 76.77% F1
    8-class benchmark: 43.87% accuracy / 43.41% F1

Usage
-----
    python src/train_model.py \\
        --config experiments/config_4class.yaml \\
        --output results/models/

    # Or pass data directly:
    python src/train_model.py \\
        --train_data data/processed/train_4class.parquet \\
        --test_data  data/processed/test_4class.parquet  \\
        --name       4class_benchmark
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from preprocess import get_feature_columns

log = logging.getLogger(__name__)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_split(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Parquet file and return (X, y) arrays."""
    df = pd.read_parquet(path)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df["activity"].values
    return X, y


# ─── Model training ───────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Fit a Random Forest on training data.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, 132)
    y_train : np.ndarray, shape (N,)
    params  : dict
        Keys: n_estimators, max_depth, min_samples_split, min_samples_leaf
    """
    rf = RandomForestClassifier(
        n_estimators      = params.get("n_estimators",       100),
        max_depth         = params.get("max_depth",           None),
        min_samples_split = params.get("min_samples_split",   2),
        min_samples_leaf  = params.get("min_samples_leaf",    1),
        random_state      = random_state,
        n_jobs            = -1,
        class_weight      = "balanced",
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f s.", elapsed)
    return rf


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Return accuracy, weighted F1, and per-class classification report."""
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    report   = classification_report(y_test, y_pred, digits=4)
    n_classes = len(np.unique(y_test))
    chance   = 1.0 / n_classes

    log.info("Accuracy : %.4f  (chance = %.4f,  gain = +%.4f)", accuracy, chance, accuracy - chance)
    log.info("F1-Score : %.4f", f1)
    print("\nClassification Report:\n", report)

    return {
        "accuracy":    float(accuracy),
        "f1_score":    float(f1),
        "chance":      float(chance),
        "improvement": float(accuracy - chance),
        "n_classes":   n_classes,
        "n_test":      len(y_test),
        "report":      report,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Random Forest for pose-based HAR.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",     type=str, help="YAML config file path.")
    group.add_argument("--train_data", type=str, help="Path to train Parquet file.")

    p.add_argument("--test_data", type=str, help="Path to test Parquet file (required with --train_data).")
    p.add_argument("--name",      type=str, default="model", help="Experiment name.")
    p.add_argument("--output",    type=str, default="results/models/", help="Output directory.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()

    # ── Resolve config ────────────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        name       = cfg["name"]
        params     = cfg.get("model_params", {})
        train_path = Path(cfg["data_path"]) / f"train_{name}.parquet"
        test_path  = Path(cfg["data_path"]) / f"test_{name}.parquet"
        random_state = cfg.get("training", {}).get("random_state", 42)
    else:
        if not args.test_data:
            raise ValueError("--test_data is required when using --train_data.")
        name         = args.name
        params       = {}
        train_path   = Path(args.train_data)
        test_path    = Path(args.test_data)
        random_state = 42

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading training data from %s", train_path)
    X_train, y_train = load_split(train_path)
    log.info("Loading test data from %s", test_path)
    X_test, y_test   = load_split(test_path)

    log.info(
        "Train: %d samples | Test: %d samples | Features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Training Random Forest (%d trees)...", params.get("n_estimators", 100))
    model = train_random_forest(X_train, y_train, params, random_state)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = output_path / f"rf_{name}.pkl"
    joblib.dump(model, model_path)
    log.info("Model saved → %s", model_path)

    # ── Save metrics ──────────────────────────────────────────────────────────
    results_path = output_path / f"results_{name}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", results_path)


if __name__ == "__main__":
    main()

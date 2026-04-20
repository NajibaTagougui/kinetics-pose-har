"""
evaluate.py
-----------
Load a trained Random Forest and produce:
  • Console classification report
  • Confusion matrix PNG (publication-ready, 300 DPI)
  • Per-class accuracy CSV
  • Summary JSON

Reproduces the figures in:
    "Privacy-Preserving Human Action Recognition for Ambient Intelligence:
     A 3D Pose-Based Benchmark"

Usage
-----
    python src/evaluate.py \\
        --model     results/models/rf_4class_benchmark.pkl \\
        --test_data data/processed/test_4class.parquet \\
        --output    results/evaluation/

    # Reproduce all paper benchmarks at once:
    python src/evaluate.py --all
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

from preprocess import get_feature_columns

log = logging.getLogger(__name__)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Save a publication-ready confusion matrix as a PNG.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground-truth and predicted labels.
    class_names : list[str]
        Ordered list of class labels.
    title : str
        Figure title.
    save_path : Path
        Output PNG path.
    dpi : int
        Resolution (default 300 for publication).
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    n  = len(class_names)
    fig_size = max(8, n * 1.5)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True",      fontsize=12, labelpad=10)
    ax.set_title(title,        fontsize=14, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0,  fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved → %s", save_path)


def plot_per_class_accuracy(
    class_names: list[str],
    accuracies: list[float],
    overall_acc: float,
    chance_level: float,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Bar chart of per-class accuracy with overall and chance baselines."""
    colours = ["#2ca02c" if a >= overall_acc else "#ff7f0e" for a in accuracies]

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.8), 6))
    bars = ax.bar(class_names, [a * 100 for a in accuracies], color=colours, edgecolor="white")

    ax.axhline(overall_acc * 100, color="#1f77b4", linestyle="--", linewidth=1.5,
               label=f"Overall: {overall_acc*100:.1f}%")
    ax.axhline(chance_level * 100, color="#d62728", linestyle=":",  linewidth=1.5,
               label=f"Chance: {chance_level*100:.1f}%")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc*100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.legend(fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Per-class accuracy chart saved → %s", save_path)


# ─── Evaluation pipeline ──────────────────────────────────────────────────────

def run_evaluation(
    model_path: Path,
    test_path: Path,
    output_dir: Path,
    name: str | None = None,
) -> dict:
    """
    Full evaluation pipeline for a single model + test-set pair.

    Returns a dict with accuracy, f1, per-class stats, and file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = name or model_path.stem

    # Load
    log.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    log.info("Loading test data from %s", test_path)
    df = pd.read_parquet(test_path)
    feature_cols = get_feature_columns(df)
    X_test = df[feature_cols].values.astype(np.float32)
    y_test = df["activity"].values

    class_names = sorted(model.classes_)
    n_classes   = len(class_names)
    chance      = 1.0 / n_classes

    # Predict
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    report   = classification_report(y_test, y_pred, labels=class_names, digits=4)

    log.info("Accuracy: %.4f  |  F1: %.4f  |  Improvement: +%.4f", accuracy, f1, accuracy - chance)
    print(report)

    # Per-class accuracy
    per_class = {}
    for cls in class_names:
        mask = y_test == cls
        per_class[cls] = {
            "accuracy":     float(accuracy_score(y_test[mask], y_pred[mask])),
            "test_samples": int(mask.sum()),
        }

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_title = (
        f"Confusion Matrix — {n_classes} Classes\n"
        f"Accuracy: {accuracy*100:.2f}%"
    )
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        title=cm_title,
        save_path=output_dir / f"confusion_{tag}.png",
    )

    # ── Per-class bar chart ───────────────────────────────────────────────────
    plot_per_class_accuracy(
        class_names,
        [per_class[c]["accuracy"] for c in class_names],
        overall_acc=accuracy,
        chance_level=chance,
        save_path=output_dir / f"per_class_{tag}.png",
    )

    # ── Save per-class CSV ────────────────────────────────────────────────────
    csv_path = output_dir / f"per_class_{tag}.csv"
    pd.DataFrame(per_class).T.to_csv(csv_path)
    log.info("Per-class results saved → %s", csv_path)

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "name":        tag,
        "accuracy":    float(accuracy),
        "f1_score":    float(f1),
        "chance":      float(chance),
        "improvement": float(accuracy - chance),
        "n_classes":   n_classes,
        "n_test":      len(y_test),
        "per_class":   per_class,
    }
    json_path = output_dir / f"summary_{tag}.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    log.info("Summary saved → %s", json_path)

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained HAR model.")
    p.add_argument("--model",     type=str, help="Path to .pkl model file.")
    p.add_argument("--test_data", type=str, help="Path to test Parquet file.")
    p.add_argument("--output",    type=str, default="results/evaluation/",
                   help="Output directory for figures and reports.")
    p.add_argument("--all",       action="store_true",
                   help="Reproduce all paper benchmarks (4-class and 8-class).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()
    output_path = Path(args.output)

    if args.all:
        benchmarks = [
            ("results/models/rf_4class_benchmark.pkl", "data/processed/test_4class.parquet",  "4class"),
            ("results/models/rf_8class_benchmark.pkl", "data/processed/test_8class.parquet",  "8class"),
        ]
        for model_p, test_p, name in benchmarks:
            log.info("=== Benchmark: %s ===", name)
            run_evaluation(Path(model_p), Path(test_p), output_path, name)
    else:
        if not args.model or not args.test_data:
            raise ValueError("Provide --model and --test_data, or use --all.")
        run_evaluation(Path(args.model), Path(args.test_data), output_path)


if __name__ == "__main__":
    main()

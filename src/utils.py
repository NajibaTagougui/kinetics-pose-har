"""
utils.py
--------
Shared plotting helpers and dataset statistics utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ─── Publication-ready plot defaults ─────────────────────────────────────────

PALETTE = sns.color_palette("muted")

plt.rcParams.update({
    "font.family":    "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ─── Feature importance ───────────────────────────────────────────────────────

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

FEATURE_COLUMNS = [
    f"{name}_{axis}"
    for name in LANDMARK_NAMES
    for axis in ("x", "y", "z", "v")
]


def plot_feature_importance(
    importances: np.ndarray,
    top_k: int = 15,
    save_path: Path | None = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Bar chart of the top-k most important pose features (Gini importance).

    Parameters
    ----------
    importances : np.ndarray, shape (132,)
        Feature importances from RandomForest.feature_importances_.
    top_k : int
        Number of top features to display.
    save_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    pd.DataFrame
        Top-k features sorted by importance.
    """
    series = pd.Series(importances, index=FEATURE_COLUMNS)
    top    = series.nlargest(top_k).sort_values()

    fig, ax = plt.subplots(figsize=(9, top_k * 0.45 + 1))
    top.plot(kind="barh", ax=ax, color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title(f"Top {top_k} Most Discriminative Pose Features", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return top.reset_index().rename(columns={"index": "feature", 0: "importance"})


# ─── Dataset statistics ───────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> None:
    """Print a summary table of the dataset."""
    print(f"{'Total frames':30s}: {len(df):>10,}")
    print(f"{'Unique videos':30s}: {df['video_id'].nunique():>10,}")
    print(f"{'Unique classes':30s}: {df['activity'].nunique():>10,}")
    print()

    vis_cols = [c for c in df.columns if c.endswith("_v")]
    if vis_cols:
        mean_vis = df[vis_cols].values.mean()
        print(f"{'Mean landmark visibility':30s}: {mean_vis*100:>9.1f}%")

    print("\nTop 10 classes by frame count:")
    top10 = df["activity"].value_counts().head(10)
    for i, (cls, cnt) in enumerate(top10.items(), 1):
        print(f"  {i:2d}. {cls:<35s}: {cnt:>10,}")


def frames_per_video_histogram(
    df: pd.DataFrame,
    save_path: Path | None = None,
    dpi: int = 300,
) -> None:
    """Histogram of frame counts per video."""
    counts = df.groupby("video_id").size()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=50, color=PALETTE[1], edgecolor="white")
    ax.axvline(counts.mean(),   color="red",    linestyle="--", label=f"Mean={counts.mean():.0f}")
    ax.axvline(counts.median(), color="orange", linestyle=":",  label=f"Median={counts.median():.0f}")
    ax.set_xlabel("Frames per video")
    ax.set_ylabel("Number of videos")
    ax.set_title("Frame Count Distribution Across Videos")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

"""
preprocess.py
-------------
Load, filter, and split the pose Parquet dataset for model training.

Key design choice — **video-level splitting**:
    All frames from a given video go to either train or test, never both.
    This prevents temporal data leakage that would artificially inflate accuracy.

Usage
-----
    python src/preprocess.py \\
        --poses_dir  data/processed/poses \\
        --output_dir data/processed \\
        --split_strategy video_level \\
        --test_size 0.2 \\
        --benchmark 4class          # or 8class / full
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── Class lists ──────────────────────────────────────────────────────────────

CLASSES_4 = [
    "archery",
    "clean and jerk",
    "crawling baby",
    "brushing teeth",
]

CLASSES_8 = [
    "air drumming",
    "archery",
    "arm wrestling",
    "blowing glass",
    "blowing out candles",
    "brushing teeth",
    "clean and jerk",
    "crawling baby",
]

BENCHMARK_CLASSES = {
    "4class": CLASSES_4,
    "8class": CLASSES_8,
    "full":   None,   # None → use all classes
}


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_poses(poses_dir: str | Path) -> pd.DataFrame:
    """
    Concatenate all per-video Parquet files into a single DataFrame.

    Expected columns: 132 feature columns + video_id + frame_index + activity.
    """
    poses_dir = Path(poses_dir)
    files = sorted(poses_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {poses_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    log.info("Loaded %d frames from %d files.", len(df), len(files))
    return df


# ─── Filtering ────────────────────────────────────────────────────────────────

def filter_classes(df: pd.DataFrame, classes: list[str] | None) -> pd.DataFrame:
    """Keep only rows belonging to the requested classes."""
    if classes is None:
        return df
    mask = df["activity"].isin(classes)
    out = df[mask].copy()
    log.info("Filtered to %d classes → %d frames.", len(classes), len(out))
    return out


def filter_visibility(df: pd.DataFrame, min_visibility: float = 0.0) -> pd.DataFrame:
    """
    Optionally drop frames whose mean landmark visibility falls below a threshold.
    Set min_visibility=0.0 (default) to keep all frames.
    """
    if min_visibility <= 0.0:
        return df
    vis_cols = [c for c in df.columns if c.endswith("_v")]
    mean_vis = df[vis_cols].mean(axis=1)
    out = df[mean_vis >= min_visibility].copy()
    log.info("Visibility filter (>= %.2f): %d → %d frames.", min_visibility, len(df), len(out))
    return out


# ─── Splitting ────────────────────────────────────────────────────────────────

def split_by_video(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Video-level train/test split — no frame from the same video appears in both.

    Stratified by activity: the test fraction is drawn proportionally from
    each class's video pool.
    """
    rng = np.random.default_rng(random_state)
    train_ids, test_ids = [], []

    for activity, group in df.groupby("activity"):
        vids = group["video_id"].unique()
        rng.shuffle(vids)
        n_test = max(1, int(len(vids) * test_size))
        test_ids.extend(vids[:n_test])
        train_ids.extend(vids[n_test:])

    train_df = df[df["video_id"].isin(train_ids)].copy()
    test_df  = df[df["video_id"].isin(test_ids)].copy()

    log.info(
        "Split → train: %d frames (%d videos) | test: %d frames (%d videos)",
        len(train_df), len(train_ids),
        len(test_df),  len(test_ids),
    )
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the 132 pose feature columns (exclude metadata)."""
    meta = {"video_id", "frame_index", "activity"}
    return [c for c in df.columns if c not in meta]


# ─── CLI entry point ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess pose data for training.")
    p.add_argument("--poses_dir",       type=str, default="data/processed/poses",
                   help="Directory of per-video Parquet files.")
    p.add_argument("--output_dir",      type=str, default="data/processed",
                   help="Where to write train/test Parquet files.")
    p.add_argument("--benchmark",       type=str, default="4class",
                   choices=["4class", "8class", "full"],
                   help="Which class subset to use.")
    p.add_argument("--split_strategy",  type=str, default="video_level",
                   choices=["video_level"],
                   help="Splitting strategy (only video_level is supported).")
    p.add_argument("--test_size",       type=float, default=0.2,
                   help="Fraction of videos to hold out for testing.")
    p.add_argument("--min_visibility",  type=float, default=0.0,
                   help="Minimum mean landmark visibility (0=no filter).")
    p.add_argument("--random_state",    type=int,   default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_poses(args.poses_dir)
    df = filter_classes(df, BENCHMARK_CLASSES[args.benchmark])
    df = filter_visibility(df, args.min_visibility)

    train_df, test_df = split_by_video(df, args.test_size, args.random_state)

    tag = args.benchmark
    train_df.to_parquet(output_path / f"train_{tag}.parquet", index=False)
    test_df.to_parquet( output_path / f"test_{tag}.parquet",  index=False)
    log.info("Saved train_%s.parquet and test_%s.parquet to %s", tag, tag, output_path)


if __name__ == "__main__":
    main()

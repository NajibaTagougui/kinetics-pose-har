"""
tests/test_preprocessing.py
----------------------------
Unit tests for data preprocessing and splitting functions.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from preprocess import (
    CLASSES_4,
    CLASSES_8,
    filter_classes,
    filter_visibility,
    get_feature_columns,
    split_by_video,
)


def make_dummy_df(n_videos: int = 20, frames_per_video: int = 10) -> pd.DataFrame:
    """Create a small synthetic pose DataFrame for testing."""
    rng = np.random.default_rng(0)
    activities = CLASSES_4
    rows = []
    for vid_idx in range(n_videos):
        activity = activities[vid_idx % len(activities)]
        for frame in range(frames_per_video):
            row = {f"lm{i}_{ax}": rng.random()
                   for i in range(33) for ax in ("x", "y", "z", "v")}
            row["video_id"]    = f"vid_{vid_idx:03d}"
            row["frame_index"] = frame
            row["activity"]    = activity
            rows.append(row)
    return pd.DataFrame(rows)


class TestFilterClasses:
    def test_keeps_only_requested(self):
        df = make_dummy_df()
        subset = CLASSES_4[:2]
        out = filter_classes(df, subset)
        assert set(out["activity"].unique()) == set(subset)

    def test_none_keeps_all(self):
        df = make_dummy_df()
        out = filter_classes(df, None)
        assert len(out) == len(df)


class TestFilterVisibility:
    def test_zero_keeps_all(self):
        df = make_dummy_df()
        out = filter_visibility(df, 0.0)
        assert len(out) == len(df)


class TestSplitByVideo:
    def test_no_leakage(self):
        """No video should appear in both train and test."""
        df = make_dummy_df(n_videos=20)
        train, test = split_by_video(df, test_size=0.2, random_state=42)
        train_vids = set(train["video_id"].unique())
        test_vids  = set(test["video_id"].unique())
        assert train_vids.isdisjoint(test_vids)

    def test_all_frames_accounted_for(self):
        df = make_dummy_df(n_videos=20)
        train, test = split_by_video(df, test_size=0.2, random_state=42)
        assert len(train) + len(test) == len(df)

    def test_approximate_split_ratio(self):
        df = make_dummy_df(n_videos=40)
        _, test = split_by_video(df, test_size=0.25, random_state=42)
        ratio = len(test) / len(df)
        assert 0.15 <= ratio <= 0.40, f"Unexpected test ratio: {ratio:.2f}"


class TestGetFeatureColumns:
    def test_excludes_metadata(self):
        df = make_dummy_df(n_videos=2)
        feat_cols = get_feature_columns(df)
        for meta in ("video_id", "frame_index", "activity"):
            assert meta not in feat_cols

    def test_correct_count(self):
        df = make_dummy_df(n_videos=2)
        feat_cols = get_feature_columns(df)
        assert len(feat_cols) == 33 * 4

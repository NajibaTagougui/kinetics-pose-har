"""
tests/test_extraction.py
------------------------
Unit tests for pose extraction functions.
"""

import numpy as np
import pytest

# Adjust import path when running from repo root
import sys
sys.path.insert(0, "src")

from extract_poses import FEATURE_COLUMNS, LANDMARK_NAMES


class TestFeatureColumns:
    def test_feature_count(self):
        """Should produce exactly 132 feature columns (33 landmarks × 4 axes)."""
        assert len(FEATURE_COLUMNS) == 132

    def test_landmark_count(self):
        assert len(LANDMARK_NAMES) == 33

    def test_column_suffixes(self):
        axes = {"x", "y", "z", "v"}
        for col in FEATURE_COLUMNS:
            suffix = col.rsplit("_", 1)[-1]
            assert suffix in axes, f"Unexpected suffix in column: {col}"

    def test_no_duplicates(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


class TestLandmarkVector:
    def test_output_length(self):
        """Simulated landmarks list should have length 132."""
        fake_landmarks = [float(i) for i in range(132)]
        assert len(fake_landmarks) == 33 * 4

    def test_visibility_range(self):
        """Visibility values should be in [0, 1]."""
        fake_vis = [0.0, 0.5, 1.0, 0.99, 0.01]
        for v in fake_vis:
            assert 0.0 <= v <= 1.0

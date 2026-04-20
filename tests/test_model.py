"""
tests/test_model.py
-------------------
Unit tests for the Random Forest training and evaluation pipeline.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from train_model import evaluate_model, train_random_forest


def make_dummy_data(
    n_samples: int = 200,
    n_features: int = 132,
    n_classes: int = 4,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = np.array([f"class_{i % n_classes}" for i in range(n_samples)])
    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


class TestTrainRandomForest:
    def test_returns_fitted_model(self):
        X_tr, X_te, y_tr, y_te = make_dummy_data()
        model = train_random_forest(X_tr, y_tr, {"n_estimators": 10})
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, "classes_")

    def test_feature_importances_sum_to_one(self):
        X_tr, _, y_tr, _ = make_dummy_data()
        model = train_random_forest(X_tr, y_tr, {"n_estimators": 10})
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-5

    def test_output_shape(self):
        X_tr, X_te, y_tr, _ = make_dummy_data()
        model = train_random_forest(X_tr, y_tr, {"n_estimators": 5})
        preds = model.predict(X_te)
        assert len(preds) == len(X_te)


class TestEvaluateModel:
    def test_metrics_keys_present(self):
        X_tr, X_te, y_tr, y_te = make_dummy_data()
        model = train_random_forest(X_tr, y_tr, {"n_estimators": 10})
        metrics = evaluate_model(model, X_te, y_te)
        for key in ("accuracy", "f1_score", "chance", "improvement", "n_classes", "n_test"):
            assert key in metrics

    def test_accuracy_in_range(self):
        X_tr, X_te, y_tr, y_te = make_dummy_data()
        model = train_random_forest(X_tr, y_tr, {"n_estimators": 10})
        metrics = evaluate_model(model, X_te, y_te)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_improvement_equals_accuracy_minus_chance(self):
        X_tr, X_te, y_tr, y_te = make_dummy_data()
        model = train_random_forest(X_tr, y_tr, {"n_estimators": 10})
        metrics = evaluate_model(model, X_te, y_te)
        expected = metrics["accuracy"] - metrics["chance"]
        assert abs(metrics["improvement"] - expected) < 1e-9

"""Unit tests for the DiabetesPredictor class."""

import os
import tempfile

import numpy as np
import pytest

from models.patient import Patient
from models.predictor import DiabetesPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n_samples: int = 200):
    """Return a small synthetic (X, y) pair for fast tests."""
    rng = np.random.default_rng(0)
    X = rng.uniform(
        low=[0, 50, 40, 0, 0, 15.0, 0.0, 18],
        high=[5, 200, 120, 60, 400, 50.0, 1.5, 75],
        size=(n_samples, 8),
    )
    y = (X[:, 1] > 130).astype(int)   # simple glucose-based label
    return X, y


def _make_patient():
    return Patient(2, 120, 70, 25, 80, 28.5, 0.5, 35)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDiabetesPredictorInit:
    def test_default_not_trained(self):
        p = DiabetesPredictor()
        assert not p.is_trained

    def test_repr_not_trained(self):
        p = DiabetesPredictor()
        assert "not trained" in repr(p)


class TestDiabetesPredictorTrain:
    def test_train_returns_accuracy(self):
        p = DiabetesPredictor()
        X, y = _make_synthetic_dataset()
        metrics = p.train(X, y)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_trained_flag_set_after_train(self):
        p = DiabetesPredictor()
        X, y = _make_synthetic_dataset()
        p.train(X, y)
        assert p.is_trained

    def test_repr_trained(self):
        p = DiabetesPredictor()
        X, y = _make_synthetic_dataset()
        p.train(X, y)
        assert "trained" in repr(p)
        assert "not trained" not in repr(p)


class TestDiabetesPredictorPredict:
    def setup_method(self):
        self.predictor = DiabetesPredictor()
        X, y = _make_synthetic_dataset()
        self.predictor.train(X, y)

    def test_predict_returns_dict(self):
        result = self.predictor.predict(_make_patient())
        assert isinstance(result, dict)

    def test_predict_keys(self):
        result = self.predictor.predict(_make_patient())
        assert "prediction" in result
        assert "probability" in result
        assert "label" in result

    def test_predict_binary_output(self):
        result = self.predictor.predict(_make_patient())
        assert result["prediction"] in (0, 1)

    def test_predict_probability_range(self):
        result = self.predictor.predict(_make_patient())
        assert 0.0 <= result["probability"] <= 100.0

    def test_predict_before_train_raises(self):
        fresh = DiabetesPredictor()
        with pytest.raises(RuntimeError):
            fresh.predict(_make_patient())


class TestDiabetesPredictorPersistence:
    def test_save_and_load(self, tmp_path):
        model_path = str(tmp_path / "model.pkl")
        scaler_path = str(tmp_path / "scaler.pkl")

        p1 = DiabetesPredictor(model_path=model_path, scaler_path=scaler_path)
        X, y = _make_synthetic_dataset()
        p1.train(X, y)
        expected = p1.predict(_make_patient())
        p1.save()

        p2 = DiabetesPredictor(model_path=model_path, scaler_path=scaler_path)
        p2.load()
        actual = p2.predict(_make_patient())

        assert actual["prediction"] == expected["prediction"]

    def test_load_missing_model_raises(self, tmp_path):
        p = DiabetesPredictor(
            model_path=str(tmp_path / "missing_model.pkl"),
            scaler_path=str(tmp_path / "missing_scaler.pkl"),
        )
        with pytest.raises(FileNotFoundError):
            p.load()


class TestDiabetesPredictorFeatureImportances:
    def test_feature_importances_keys(self):
        p = DiabetesPredictor()
        X, y = _make_synthetic_dataset()
        p.train(X, y)
        importances = p.feature_importances()
        assert set(importances.keys()) == set(DiabetesPredictor.FEATURE_NAMES)

    def test_feature_importances_sum_to_one(self):
        p = DiabetesPredictor()
        X, y = _make_synthetic_dataset()
        p.train(X, y)
        importances = p.feature_importances()
        total = sum(importances.values())
        assert abs(total - 1.0) < 1e-6

    def test_feature_importances_before_train_raises(self):
        p = DiabetesPredictor()
        with pytest.raises(RuntimeError):
            p.feature_importances()

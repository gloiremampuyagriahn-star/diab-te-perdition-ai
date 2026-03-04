"""Unit tests for the DiabetesApp Flask application."""

import numpy as np
import pytest

from models.predictor import DiabetesPredictor


def _make_synthetic_dataset(n_samples: int = 200):
    rng = np.random.default_rng(0)
    X = rng.uniform(
        low=[0, 50, 40, 0, 0, 15.0, 0.0, 18],
        high=[5, 200, 120, 60, 400, 50.0, 1.5, 75],
        size=(n_samples, 8),
    )
    y = (X[:, 1] > 130).astype(int)
    return X, y


@pytest.fixture()
def trained_predictor(tmp_path):
    """Return a trained predictor saved to a temporary location."""
    model_path = str(tmp_path / "model.pkl")
    scaler_path = str(tmp_path / "scaler.pkl")
    predictor = DiabetesPredictor(model_path=model_path, scaler_path=scaler_path)
    X, y = _make_synthetic_dataset()
    predictor.train(X, y)
    predictor.save()
    return model_path, scaler_path


@pytest.fixture()
def client(trained_predictor):
    from app import DiabetesApp
    model_path, scaler_path = trained_predictor
    diabetes_app = DiabetesApp(model_path=model_path, scaler_path=scaler_path)
    diabetes_app.flask_app.config["TESTING"] = True
    with diabetes_app.flask_app.test_client() as c:
        yield c


class TestIndexRoute:
    def test_get_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_get_contains_form(self, client):
        response = client.get("/")
        assert b"form" in response.data


class TestPredictRoute:
    _FORM_DATA = {
        "pregnancies": "2",
        "glucose": "120",
        "blood_pressure": "70",
        "skin_thickness": "25",
        "insulin": "80",
        "bmi": "28.5",
        "diabetes_pedigree_function": "0.5",
        "age": "35",
    }

    def test_post_returns_200(self, client):
        response = client.post("/predict", data=self._FORM_DATA)
        assert response.status_code == 200

    def test_post_returns_result(self, client):
        response = client.post("/predict", data=self._FORM_DATA)
        # Should show either "Diabétique" or "Non diabétique"
        assert b"iab" in response.data

    def test_invalid_glucose_shows_error(self, client):
        bad_data = dict(self._FORM_DATA, glucose="9999")
        response = client.post("/predict", data=bad_data)
        assert response.status_code == 200
        assert b"Erreur" in response.data or b"erreur" in response.data


class TestAboutRoute:
    def test_get_returns_200(self, client):
        response = client.get("/about")
        assert response.status_code == 200

    def test_about_contains_methodology(self, client):
        response = client.get("/about")
        assert b"Random Forest" in response.data or b"thodologie" in response.data

"""DiabetesPredictor — machine-learning predictor for diabetes classification."""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .patient import Patient

# Default path where the trained model artefacts are persisted.
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model.pkl")
_DEFAULT_SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_scaler.pkl")


class DiabetesPredictor:
    """Trains, evaluates, persists and uses a Random Forest classifier to
    predict whether a patient has diabetes.

    Parameters
    ----------
    model_path : str, optional
        File path used to save / load the trained model.
    scaler_path : str, optional
        File path used to save / load the fitted scaler.
    n_estimators : int
        Number of trees in the Random Forest (default: 100).
    random_state : int
        Seed for reproducibility (default: 42).
    """

    FEATURE_NAMES = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
    ]

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        scaler_path: str = _DEFAULT_SCALER_PATH,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """Train the classifier on the provided dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 8)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Binary target (0 = no diabetes, 1 = diabetes).
        test_size : float
            Fraction of samples reserved for evaluation (default: 0.2).

        Returns
        -------
        dict
            Dictionary with ``accuracy`` and ``report`` keys.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self._model.fit(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)

        return {"accuracy": accuracy, "report": report}

    def predict(self, patient: Patient) -> dict:
        """Predict whether the given patient has diabetes.

        Parameters
        ----------
        patient : Patient
            Patient instance whose features will be used.

        Returns
        -------
        dict
            ``{"prediction": int, "probability": float, "label": str}``

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        """
        self._ensure_ready()

        features = np.array([patient.to_list()])
        features_scaled = self._scaler.transform(features)

        prediction = int(self._model.predict(features_scaled)[0])
        probability = float(self._model.predict_proba(features_scaled)[0][prediction])
        label = "Diabétique" if prediction == 1 else "Non diabétique"

        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "label": label,
        }

    def save(self) -> None:
        """Persist the trained model and scaler to disk."""
        self._ensure_ready()
        joblib.dump(self._model, self.model_path)
        joblib.dump(self._scaler, self.scaler_path)

    def load(self) -> None:
        """Load a previously saved model and scaler from disk.

        Raises
        ------
        FileNotFoundError
            If the model or scaler file is missing.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler file not found: {self.scaler_path}"
            )
        self._model = joblib.load(self.model_path)
        self._scaler = joblib.load(self.scaler_path)

    @property
    def is_trained(self) -> bool:
        """Return ``True`` if the model and scaler are ready to use."""
        return self._model is not None and self._scaler is not None

    def feature_importances(self) -> dict:
        """Return a mapping of feature name → importance score.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        """
        self._ensure_ready()
        importances = self._model.feature_importances_
        return dict(zip(self.FEATURE_NAMES, importances.tolist()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "not trained"
        return (
            f"DiabetesPredictor(n_estimators={self.n_estimators}, "
            f"random_state={self.random_state}, status={status})"
        )

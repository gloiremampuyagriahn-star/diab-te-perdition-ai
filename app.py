"""Flask web application for diabetes prediction.

The application exposes:
- ``GET  /``         — home page with prediction form
- ``POST /predict``  — returns prediction result
- ``GET  /about``    — about / methodology page
"""

from __future__ import annotations

import os

from flask import Flask, render_template, request

from models.patient import Patient
from models.predictor import DiabetesPredictor


class DiabetesApp:
    """Encapsulates the Flask application and prediction logic.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved model file.
    scaler_path : str, optional
        Path to the saved scaler file.
    """

    def __init__(
        self,
        model_path: str | None = None,
        scaler_path: str | None = None,
    ) -> None:
        self._app = Flask(__name__)
        self._predictor = DiabetesPredictor(
            **(
                {"model_path": model_path, "scaler_path": scaler_path}
                if model_path and scaler_path
                else {}
            )
        )
        self._load_model()
        self._register_routes()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load a pre-trained model; warn if unavailable."""
        try:
            self._predictor.load()
        except FileNotFoundError:
            self._app.logger.warning(
                "No trained model found. Run train_model.py first."
            )

    def _register_routes(self) -> None:
        """Bind URL rules to handler methods."""
        self._app.add_url_rule("/", "index", self._index, methods=["GET"])
        self._app.add_url_rule(
            "/predict", "predict", self._predict, methods=["POST"]
        )
        self._app.add_url_rule("/about", "about", self._about, methods=["GET"])

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _index(self):
        return render_template("index.html")

    def _predict(self):
        form = request.form
        error = None
        result = None

        try:
            patient = Patient(
                pregnancies=int(form.get("pregnancies", 0)),
                glucose=float(form.get("glucose", 0)),
                blood_pressure=float(form.get("blood_pressure", 0)),
                skin_thickness=float(form.get("skin_thickness", 0)),
                insulin=float(form.get("insulin", 0)),
                bmi=float(form.get("bmi", 0)),
                diabetes_pedigree_function=float(
                    form.get("diabetes_pedigree_function", 0)
                ),
                age=int(form.get("age", 0)),
            )

            if not self._predictor.is_trained:
                raise RuntimeError(
                    "Le modèle n'est pas encore entraîné. "
                    "Veuillez exécuter train_model.py d'abord."
                )

            result = self._predictor.predict(patient)
        except (ValueError, RuntimeError) as exc:
            error = str(exc)

        return render_template("result.html", result=result, error=error)

    def _about(self):
        importances = None
        if self._predictor.is_trained:
            importances = self._predictor.feature_importances()
        return render_template("about.html", importances=importances)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, **kwargs) -> None:
        """Start the Flask development server."""
        self._app.run(**kwargs)

    @property
    def flask_app(self) -> Flask:
        """Expose the underlying :class:`~flask.Flask` instance."""
        return self._app

    def __repr__(self) -> str:
        return (
            f"DiabetesApp(model_trained={self._predictor.is_trained})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app = DiabetesApp()
    app.run(host="0.0.0.0", port=port, debug=True)

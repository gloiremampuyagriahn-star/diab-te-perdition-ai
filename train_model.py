"""Script to train and save the DiabetesPredictor model.

Usage
-----
    python train_model.py

The script downloads (or uses a local copy of) the Pima Indians Diabetes
dataset, trains the model, evaluates it, and saves the artefacts to disk.
"""

import os
import urllib.request

import numpy as np
import pandas as pd

from models.predictor import DiabetesPredictor

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")

COLUMN_NAMES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "outcome",
]


def load_dataset(path: str, url: str) -> pd.DataFrame:
    """Load dataset from *path*, downloading from *url* if necessary."""
    if not os.path.exists(path):
        print(f"Downloading dataset from {url} …")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    df = pd.read_csv(path, names=COLUMN_NAMES)
    return df


def main() -> None:
    df = load_dataset(DATASET_PATH, DATASET_URL)

    X = df[COLUMN_NAMES[:-1]].values
    y = df["outcome"].values

    predictor = DiabetesPredictor()
    print("Training model …")
    metrics = predictor.train(X, y)

    accuracy = metrics["accuracy"]
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    predictor.save()
    print(
        f"Model saved to '{predictor.model_path}'\n"
        f"Scaler saved to '{predictor.scaler_path}'"
    )


if __name__ == "__main__":
    main()

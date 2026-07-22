from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


SPECIES_NAMES = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica",
]


def load_model(model_path: str | Path):
    """
    Load the trained machine learning model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model was not found. Run train_model.py first."
        )

    return joblib.load(model_path)


def load_metrics(metrics_path: str | Path) -> dict:
    """
    Load model evaluation metrics from JSON.
    """

    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(
            "Metrics file was not found. Run train_model.py first."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def predict_species(
    model,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> dict:
    """
    Predict the Iris species and return class probabilities.
    """

    features = np.array(
        [
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            ]
        ],
        dtype=float,
    )

    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    probability_results = {
        SPECIES_NAMES[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }

    return {
        "class_id": predicted_class,
        "species": SPECIES_NAMES[predicted_class],
        "probabilities": probability_results,
    }
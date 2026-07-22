from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "knn_iris_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"


def load_dataset():
    """
    Load the Iris dataset as a Pandas DataFrame.
    """

    iris = load_iris(as_frame=True)

    features = iris.data
    target = iris.target

    return iris, features, target


def train_model() -> None:
    """
    Train and evaluate a K-Nearest Neighbors classification model.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    iris, features, target = load_dataset()

    print("=" * 60)
    print("IRIS INTELLIGENCE CLASSIFIER")
    print("=" * 60)

    print("\nDataset shape:")
    print(features.shape)

    print("\nFeature names:")
    print(list(features.columns))

    print("\nTarget classes:")
    print(list(iris.target_names))

    print("\nClass distribution:")
    print(target.value_counts().sort_index())

    # 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
        shuffle=True,
    )

    print("\nTraining samples:", len(x_train))
    print("Testing samples:", len(x_test))

    # StandardScaler prevents large-scale features
    # from dominating distance calculations in KNN.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    # Try multiple K values and select the best one.
    parameter_grid = {
        "knn__n_neighbors": list(range(1, 16, 2))
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )

    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "dataset_name": "Iris Dataset",
        "total_samples": int(len(features)),
        "training_samples": int(len(x_train)),
        "testing_samples": int(len(x_test)),
        "number_of_features": int(features.shape[1]),
        "number_of_classes": int(len(iris.target_names)),
        "feature_names": list(features.columns),
        "target_names": list(iris.target_names),
        "best_k": int(
            search.best_params_["knn__n_neighbors"]
        ),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    # Save the trained pipeline.
    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    # Save metrics.
    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save actual and predicted test values.
    prediction_results = x_test.copy()

    prediction_results["actual_class"] = (
        y_test.values
    )

    prediction_results["predicted_class"] = (
        predictions
    )

    prediction_results["actual_species"] = [
        iris.target_names[value]
        for value in y_test.values
    ]

    prediction_results["predicted_species"] = [
        iris.target_names[value]
        for value in predictions
    ]

    prediction_results.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nBest K value:")
    print(metrics["best_k"])

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nPrecision:")
    print(f"{precision:.4f}")

    print("\nRecall:")
    print(f"{recall:.4f}")

    print("\nF1 Score:")
    print(f"{f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            zero_division=0,
        )
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nMetrics saved to:")
    print(METRICS_PATH)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_model()
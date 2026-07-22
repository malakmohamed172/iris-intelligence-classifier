from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_service import predict_species


def create_test_model():
    iris = load_iris()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5
                ),
            ),
        ]
    )

    model.fit(
        iris.data,
        iris.target,
    )

    return model


def test_prediction_returns_valid_species():
    model = create_test_model()

    result = predict_species(
        model=model,
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
    )

    assert result["species"] in [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica",
    ]


def test_probabilities_sum_to_one():
    model = create_test_model()

    result = predict_species(
        model=model,
        sepal_length=6.2,
        sepal_width=2.8,
        petal_length=4.8,
        petal_width=1.8,
    )

    probability_sum = sum(
        result["probabilities"].values()
    )

    assert abs(probability_sum - 1.0) < 0.0001
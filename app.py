from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris

from src.model_service import (
    load_metrics,
    load_model,
    predict_species,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "knn_iris_model.joblib"
)

METRICS_PATH = (
    BASE_DIR
    / "models"
    / "model_metrics.json"
)


st.set_page_config(
    page_title="Iris Intelligence Lab",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(126, 217, 87, 0.25);
        }

        .hero {
            padding: 1.5rem;
            border-radius: 20px;
            margin-bottom: 1rem;
            border: 1px solid rgba(126, 217, 87, 0.25);
            background:
                linear-gradient(
                    135deg,
                    rgba(64, 145, 108, 0.18),
                    rgba(52, 152, 219, 0.12)
                );
        }

        .hero h1 {
            margin-bottom: 0.2rem;
        }

        .hero p {
            opacity: 0.85;
        }

        .status {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: rgba(46, 204, 113, 0.15);
            border: 1px solid rgba(46, 204, 113, 0.35);
            font-size: 0.8rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics():
    return load_metrics(METRICS_PATH)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)

    dataframe = iris.frame.copy()

    dataframe["species"] = dataframe[
        "target"
    ].map(
        {
            index: name.title()
            for index, name
            in enumerate(iris.target_names)
        }
    )

    return dataframe


def render_home(metrics: dict) -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="status">● MODEL READY</span>
            <h1>🌸 Iris Intelligence Lab</h1>
            <p>
                An interactive supervised-learning system
                for Iris flower classification using K-Nearest Neighbors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Dataset Samples",
        metrics["total_samples"],
    )

    column2.metric(
        "Features",
        metrics["number_of_features"],
    )

    column3.metric(
        "Classes",
        metrics["number_of_classes"],
    )

    column4.metric(
        "Best K",
        metrics["best_k"],
    )

    st.subheader("Project Overview")

    st.markdown(
        """
        Iris Intelligence Lab classifies Iris flowers into:

        - Iris Setosa
        - Iris Versicolor
        - Iris Virginica

        The prediction is based on four flower measurements:

        - Sepal length
        - Sepal width
        - Petal length
        - Petal width
        """
    )

    st.subheader("Machine Learning Pipeline")

    st.code(
        """
Iris Dataset
    ↓
Data Exploration
    ↓
Train-Test Split (80% / 20%)
    ↓
StandardScaler
    ↓
K-Nearest Neighbors
    ↓
K Value Tuning
    ↓
Prediction
    ↓
Accuracy, Precision, Recall, F1 Score
    ↓
Confusion Matrix
        """,
        language="text",
    )


def render_dataset_explorer(
    dataframe: pd.DataFrame,
) -> None:
    st.title("📊 Dataset Explorer")

    st.write(
        "Explore the Iris dataset used to train "
        "the classification model."
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dataset Summary")

    st.dataframe(
        dataframe.describe(),
        use_container_width=True,
    )

    chart_type = st.selectbox(
        "Choose visualization",
        [
            "Petal Length vs Petal Width",
            "Sepal Length vs Sepal Width",
            "Class Distribution",
            "Feature Distribution",
        ],
    )

    if chart_type == "Petal Length vs Petal Width":
        figure = px.scatter(
            dataframe,
            x="petal length (cm)",
            y="petal width (cm)",
            color="species",
            title="Petal Measurements by Species",
            hover_data=[
                "sepal length (cm)",
                "sepal width (cm)",
            ],
        )

    elif chart_type == "Sepal Length vs Sepal Width":
        figure = px.scatter(
            dataframe,
            x="sepal length (cm)",
            y="sepal width (cm)",
            color="species",
            title="Sepal Measurements by Species",
            hover_data=[
                "petal length (cm)",
                "petal width (cm)",
            ],
        )

    elif chart_type == "Class Distribution":
        distribution = (
            dataframe["species"]
            .value_counts()
            .rename_axis("species")
            .reset_index(name="count")
        )

        figure = px.bar(
            distribution,
            x="species",
            y="count",
            title="Iris Class Distribution",
        )

    else:
        selected_feature = st.selectbox(
            "Select feature",
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
        )

        figure = px.histogram(
            dataframe,
            x=selected_feature,
            color="species",
            barmode="overlay",
            title=f"Distribution of {selected_feature}",
        )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def render_model_evaluation(
    metrics: dict,
) -> None:
    st.title("📈 Model Evaluation")

    columns = st.columns(4)

    columns[0].metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%",
    )

    columns[1].metric(
        "Precision",
        f"{metrics['precision_macro'] * 100:.2f}%",
    )

    columns[2].metric(
        "Recall",
        f"{metrics['recall_macro'] * 100:.2f}%",
    )

    columns[3].metric(
        "F1 Score",
        f"{metrics['f1_macro'] * 100:.2f}%",
    )

    st.subheader("Confusion Matrix")

    confusion_dataframe = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[
            name.title()
            for name in metrics["target_names"]
        ],
        columns=[
            name.title()
            for name in metrics["target_names"]
        ],
    )

    confusion_figure = px.imshow(
        confusion_dataframe,
        text_auto=True,
        labels={
            "x": "Predicted Species",
            "y": "Actual Species",
            "color": "Number of Samples",
        },
        title="KNN Confusion Matrix",
    )

    st.plotly_chart(
        confusion_figure,
        use_container_width=True,
    )

    st.subheader("Classification Report")

    report = metrics[
        "classification_report"
    ]

    report_rows = []

    for species in metrics["target_names"]:
        species_metrics = report[species]

        report_rows.append(
            {
                "Species": species.title(),
                "Precision": species_metrics[
                    "precision"
                ],
                "Recall": species_metrics[
                    "recall"
                ],
                "F1 Score": species_metrics[
                    "f1-score"
                ],
                "Support": species_metrics[
                    "support"
                ],
            }
        )

    report_dataframe = pd.DataFrame(
        report_rows
    )

    st.dataframe(
        report_dataframe.style.format(
            {
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}",
                "Support": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"The selected optimal K value is "
        f"{metrics['best_k']}."
    )


def render_prediction(
    model,
    dataframe: pd.DataFrame,
) -> None:
    st.title("🔮 Flower Species Prediction")

    st.write(
        "Enter the four flower measurements "
        "to classify a new Iris sample."
    )

    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    defaults = {
        feature: float(dataframe[feature].mean())
        for feature in feature_columns
    }

    minimums = {
        feature: float(dataframe[feature].min())
        for feature in feature_columns
    }

    maximums = {
        feature: float(dataframe[feature].max())
        for feature in feature_columns
    }

    column1, column2 = st.columns(2)

    with column1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=minimums[
                "sepal length (cm)"
            ],
            max_value=maximums[
                "sepal length (cm)"
            ],
            value=defaults[
                "sepal length (cm)"
            ],
            step=0.1,
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=minimums[
                "sepal width (cm)"
            ],
            max_value=maximums[
                "sepal width (cm)"
            ],
            value=defaults[
                "sepal width (cm)"
            ],
            step=0.1,
        )

    with column2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=minimums[
                "petal length (cm)"
            ],
            max_value=maximums[
                "petal length (cm)"
            ],
            value=defaults[
                "petal length (cm)"
            ],
            step=0.1,
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=minimums[
                "petal width (cm)"
            ],
            max_value=maximums[
                "petal width (cm)"
            ],
            value=defaults[
                "petal width (cm)"
            ],
            step=0.1,
        )

    if st.button(
        "Classify Flower",
        type="primary",
        use_container_width=True,
    ):
        result = predict_species(
            model=model,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )

        st.success(
            f"Predicted Species: "
            f"**{result['species']}**"
        )

        probability_dataframe = pd.DataFrame(
            {
                "Species": list(
                    result[
                        "probabilities"
                    ].keys()
                ),
                "Probability": [
                    probability * 100
                    for probability
                    in result[
                        "probabilities"
                    ].values()
                ],
            }
        )

        probability_figure = px.bar(
            probability_dataframe,
            x="Species",
            y="Probability",
            title="Prediction Probabilities",
            text_auto=".2f",
        )

        probability_figure.update_yaxes(
            range=[0, 100],
            title="Probability (%)",
        )

        st.plotly_chart(
            probability_figure,
            use_container_width=True,
        )

        st.dataframe(
            probability_dataframe.style.format(
                {
                    "Probability": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_about() -> None:
    st.title("ℹ️ About the Project")

    st.markdown(
        """
        **Iris Intelligence Lab** is a supervised-learning
        classification project developed for DecodeLabs Project 2.

        The project demonstrates:

        - Dataset loading and understanding
        - Exploratory data analysis
        - Train-test splitting
        - Feature scaling
        - K-Nearest Neighbors classification
        - Hyperparameter tuning
        - Predictions on unseen data
        - Accuracy, precision, recall, and F1 evaluation
        - Confusion-matrix analysis
        - Interactive Streamlit deployment
        """
    )


add_custom_css()

try:
    model = get_model()
    metrics = get_metrics()
    dataframe = get_dataset()

except FileNotFoundError:
    st.error(
        "The trained model files are missing."
    )

    st.code(
        "python train_model.py",
        language="powershell",
    )

    st.stop()


with st.sidebar:
    st.markdown("## 🌸 Iris Intelligence")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Dataset Explorer",
            "Model Evaluation",
            "Predict Species",
            "About",
        ],
    )

    st.divider()

    st.caption(
        "DecodeLabs Project 2 — "
        "Data Classification Using AI"
    )


if page == "Home":
    render_home(metrics)

elif page == "Dataset Explorer":
    render_dataset_explorer(
        dataframe
    )

elif page == "Model Evaluation":
    render_model_evaluation(
        metrics
    )

elif page == "Predict Species":
    render_prediction(
        model,
        dataframe,
    )

else:
    render_about()
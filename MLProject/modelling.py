from __future__ import annotations

import os
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    dagshub.init(
        repo_owner="anhartamim36",
        repo_name="Workflow_CI_Muhammad-Anhar-Tamim",
        mlflow=True,
    )

EXPERIMENT_NAME = "MSML_Breast_Cancer_Basic"


def resolve_data_path(filename: str) -> Path:
    candidate = BASE_DIR / "breast_cancer_preprocessing" / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find {filename} in expected locations")


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = resolve_data_path("breast_cancer_train_preprocessed.csv")
    test_path = resolve_data_path("breast_cancer_test_preprocessed.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df, test_df = load_dataset()
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))

    mlflow.sklearn.log_model(
        sk_model=model, artifact_path="model", input_example=X_test.iloc[:5]
    )

    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
    mlflow.log_dict(
        {"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()},
        "confusion_matrix.json",
    )

    print("Basic training finished successfully!")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
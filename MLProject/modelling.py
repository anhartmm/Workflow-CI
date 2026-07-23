from __future__ import annotations

import os
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
dagshub.init(repo_owner='anhartamim36', repo_name='Workflow_CI_Muhammad-Anhar-Tamim', mlflow=True)
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
    in_project_run = bool(os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_PARENT_RUN_ID"))
    run_id = os.environ.get("MLFLOW_RUN_ID") if in_project_run else None
    if not in_project_run:
        mlflow.sklearn.autolog(log_models=True)

    train_df, test_df = load_dataset()
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model = LogisticRegression(max_iter=1000, random_state=42)

    run_context = mlflow.start_run(run_name="logreg_basic") if not in_project_run else None
    with run_context if run_context is not None else nullcontext():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        for key, value in metrics.items():
            mlflow.log_metric(key, float(value), run_id=run_id)

        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt", run_id=run_id)
        mlflow.log_dict(
            {"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()},
            "confusion_matrix.json",
            run_id=run_id,
        )

        print("Basic training finished")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()

"""
modelling.py (MLflow Project version)
======================================
Pelatihan model Random Forest untuk Customer Churn dengan MLflow autolog.
Mendukung argumen CLI untuk integrasi MLflow Project & GitHub Actions CI.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Train RF model for Customer Churn")
    parser.add_argument("--dataset_path", type=str,
                        default="Customer-Churn-Records_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.dataset_path)
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )
    print(f"[INFO] Dataset: {df.shape}, Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_experiment("Customer-Churn-CI")
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)

        print(f"\n=== Evaluation Metrics ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print(classification_report(y_test, y_pred))

    print("[DONE] Training selesai. Artefak tersimpan di MLflow.")


if __name__ == "__main__":
    main()

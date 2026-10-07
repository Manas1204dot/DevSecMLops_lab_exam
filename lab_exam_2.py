import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def build_dataset(n=1500, seed=42):
    """Create a synthetic but realistic customer-churn dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "num_services": rng.integers(1, 6, n),
        "support_calls": rng.poisson(2, n),
        "is_senior": rng.integers(0, 2, n),
    })

    logit = (
        0.06 * df.support_calls
        - 0.04 * df.tenure
        + 0.015 * df.monthly_charges
        + 0.3 * df.is_senior
        - 1.0
    )
    prob = 1 / (1 + np.exp(-logit))
    df["churn"] = (rng.random(n) < prob).astype(int)
    return df


def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "f1_score": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }


def main():
    df = build_dataset()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    mlflow.set_experiment("customer-churn")

    candidates = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), True),
        "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=42), False),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), False),
    }

    print("=== Churn Prediction Experiments (MLflow) ===")
    print(f"{'Model':22s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s}")
    results = {}

    for name, (model, needs_scaling) in candidates.items():
        with mlflow.start_run(run_name=name):
            Xtr = X_tr_s if needs_scaling else X_tr.values
            Xte = X_te_s if needs_scaling else X_te.values
            model.fit(Xtr, y_tr)
            metrics = evaluate(model, Xte, y_te)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            print(f"{name:22s} {metrics['accuracy']:6.3f} {metrics['f1_score']:6.3f} {metrics['roc_auc']:6.3f}")
            results[name] = metrics

    best_model, best_metrics = max(results.items(), key=lambda kv: kv[1]["roc_auc"])
    print(f"=== Best model by ROC-AUC: {best_model} ({best_metrics['roc_auc']:.3f}) ===")
    print("=== # Inspect and compare all runs in the browser: $ mlflow ui ===")
    print("[INFO] Listening at: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
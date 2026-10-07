import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class DataPipeline:
    """A minimal but complete automated ML data pipeline."""

    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        os.makedirs(artifact_dir, exist_ok=True)

    # ---- Stage 1: Extract ----
    def extract(self):
        data = load_iris(as_frame=True)
        df = data.frame
        print(f"[EXTRACT ] loaded {df.shape[0]} rows x {df.shape[1]} cols")
        return df

    # ---- Stage 2: Validate ----
    def validate(self, df):
        missing = int(df.isnull().sum().sum())
        assert missing == 0, "Data contains missing values!"
        assert "target" in df.columns, "Target column missing!"
        print("[VALIDATE] no missing values, target present -> OK")
        return df

    # ---- Stage 3: Transform ----
    def transform(self, df):
        X = df.drop(columns=["target"])
        y = df["target"]
        X_scaled = self.scaler.fit_transform(X)
        print(f"[TRANSFORM] standard-scaled {X.shape[1]} features")
        return X_scaled, y

    # ---- Stage 4 + 5: Train and Evaluate ----
    def train(self, X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        print(f"[TRAIN   ] model fitted, test accuracy = {acc:.3f}")
        print("[REPORT  ] per-class metrics:")
        print(classification_report(y_te, y_pred, digits=2))
        return acc

    # ---- Stage 6: Persist ----
    def persist(self):
        model_path = os.path.join(self.artifact_dir, "model.pkl")
        scaler_path = os.path.join(self.artifact_dir, "scaler.pkl")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"[PERSIST ] saved model -> {model_path}")
        print(f"[PERSIST ] saved scaler -> {scaler_path}")
        return model_path, scaler_path

    def run(self):
        print("=== Automated Data Pipeline Started ===")
        df = self.extract()
        df = self.validate(df)
        X_scaled, y = self.transform(df)
        acc = self.train(X_scaled, y)
        self.persist()
        print("=== Pipeline Completed Successfully ===")
        return acc


if __name__ == "__main__":
    DataPipeline().run()


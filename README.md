# DevSecMLops Lab Exam

This repository contains two small machine-learning pipeline exercises. They demonstrate preparing data, training and evaluating models, and saving or tracking experiment results.

## Objectives

- **`lab_exam.py` — Iris data pipeline:** Load the Iris dataset, validate it, scale its features, train a random forest classifier, report classification metrics, and save the trained model and scaler.
- **`lab_exam_2.py` — Churn model experiments:** Generate a synthetic customer-churn dataset, train and compare logistic regression, random forest, and gradient boosting classifiers, and log their accuracy, F1 score, and ROC-AUC with MLflow.

## Requirements

Python 3. Install the required packages from a terminal:

```powershell
python -m pip install joblib mlflow numpy pandas scikit-learn
```

## Procedure

1. Clone the repository and change to its directory:

   ```powershell
   git clone https://github.com/Manas1204dot/DevSecMLops_lab_exam.git
   cd DevSecMLops_lab_exam
   ```

2. Install the packages listed in **Requirements**.

3. Run the Iris pipeline:

   ```powershell
   python lab_exam.py
   ```

   The script prints validation, training, and classification-report results. It saves the model and scaler as `artifacts/model.pkl` and `artifacts/scaler.pkl`.

4. Run the churn experiments:

   ```powershell
   python lab_exam_2.py
   ```

   The script prints the three models' metrics and the best model by ROC-AUC. MLflow records each experiment run in its local tracking store.

5. To inspect the MLflow runs in a browser, start the UI from the repository directory:

   ```powershell
   mlflow ui
   ```

   Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

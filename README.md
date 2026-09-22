# Telco Customer Churn Prediction – MLOps

## 📌 Project Overview

This project implements a machine learning pipeline to predict whether a telecom customer is likely to churn.

The project demonstrates an end-to-end MLOps workflow including:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature scaling and encoding
- Random Forest classification
- Model evaluation
- Git-based version control
- MLflow experiment tracking
- Reproducibility validation

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze the Telco Customer Churn dataset.
2. Clean and preprocess the data.
3. Create a reproducible train-test split.
4. Train a Random Forest classification model.
5. Evaluate the model using classification metrics.
6. Track experiments using MLflow.
7. Validate model reproducibility.

---

## 📂 Project Structure

```text
churn_prediction/
│
├── data/
│   ├── raw/
│   │   └── churn.csv
│   │
│   └── processed/
│       ├── dataset_metadata.json
│       ├── X_train_final.npy
│       ├── X_test_final.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── notebooks/
│   └── project_implementation.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_mlflow.py
│   └── validate_reproducibility.py
│
├── pipelines/
│   ├── run_lab3_baseline.py
│   └── run_lab4_tracking.py
│
├── models/
│   ├── scaler.pkl
│   ├── ohe.pkl
│   ├── random_forest_baseline.pkl
│   └── random_forest_model.pkl
│
├── outputs/
│   ├── false_negatives.csv
│   └── false_positives.csv
│
├── artifacts/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── reproducibility_report.json
│
├── requirements.txt
├── .gitignore
└── README.md
```
## 👨‍💻 Author

**B. Surya Prakash**

**CSE – Artificial Intelligence and Machine Learning (AIML)**

Telco Customer Churn Prediction – MLOps Project


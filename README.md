# Telco Customer Churn Prediction – MLOps

## 📌 Project Overview

This project implements a machine learning pipeline to predict whether a telecom customer is likely to churn.

The project demonstrates an end-to-end MLOps workflow, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing and validation
- Feature scaling and categorical encoding
- Random Forest classification
- Model evaluation and quality gates
- Git-based version control using GitHub
- MLflow experiment tracking and model registry
- Model lifecycle management
- Reproducibility validation
- Automated CI workflows using GitHub Actions
- Failure-injection testing and report generation

## 🎯 Objectives

The main objectives of this project are:

1. Analyze the Telco Customer Churn dataset.
2. Clean and preprocess the data.
3. Validate dataset quality and structure.
4. Create a reproducible train-test split.
5. Train a Random Forest classification model.
6. Evaluate the model using classification metrics.
7. Track experiments using MLflow.
8. Register and manage model versions.
9. Apply quality gates to model evaluation results.
10. Automate validation and testing through GitHub Actions.
11. Generate reports for evaluation and failure tests.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Experiment Tracking:** MLflow
- **Configuration:** YAML
- **Data Validation:** Pandera
- **Model Serialization:** Joblib
- **Version Control:** Git and GitHub
- **Automation:** GitHub Actions
- **Development Environment:** VS Code, Jupyter Notebook

## 📂 Project Structure

```text
churn_pridiction/
│
├── .github/
│   └── workflows/
│       ├── lab8_ci.yml
│       └── lab9_failure_tests.yml
│
├── data/
│   ├── raw/
│   │   └── churn.csv
│   └── processed/
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
│   ├── preprocess_pipeline.py
│   ├── train.py
│   ├── train_mlflow.py
│   ├── train_registry.py
│   ├── evaluate.py
│   ├── evaluate_model.py
│   ├── validate_data.py
│   ├── validate_outputs.py
│   ├── validate_reproducibility.py
│   ├── automate_lifecycle.py
│   ├── generate_registry_report.py
│   ├── quality_gate.py
│   └── failure_injection.py
│
├── pipelines/
│   ├── lab5_pipeline.py
│   └── lab6_registry.py
│
├── models/
│   └── Generated model and preprocessing artifacts
│
├── reports/
│   ├── evaluation_report.json
│   ├── quality_gate_report.json
│   └── lab9_failure_report.json
│
├── artifacts/
│   └── Generated model registry reports
│
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Generated models and processed-data files may be excluded from Git using `.gitignore`. The structure above summarizes the main project components; some artifacts are created when the pipelines run.*

## 📊 Dataset

The project uses the Telco Customer Churn dataset stored at:

```text
data/raw/churn.csv
```

The dataset is processed to prepare numerical and categorical features for machine learning.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Surya8522/Churn-prediction.git
cd Churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv churn-mlops
```

Activate it on Windows PowerShell:

```powershell
.\churn-mlops\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the configuration

Project settings are maintained in `config.yaml`, including data paths, model parameters, MLflow settings, evaluation thresholds, and report locations.

## 🔄 MLOps Workflow

The project is organized into progressive lab stages.

| Lab | Main activity |
|---|---|
| Labs 1–2 | Dataset exploration and preprocessing foundations |
| Lab 3 | Baseline model training and evaluation |
| Lab 4 | MLflow experiment tracking |
| Lab 5 | Production data pipeline and validation |
| Lab 6 | Model registry and lifecycle management |
| Lab 7 | Model evaluation and quality-gate reporting |
| Lab 8 | Continuous integration using GitHub Actions |
| Lab 9 | Failure-injection checks and automated reports |

## 🧪 Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Run the evaluation script after the processed test data and trained model are available:

```bash
python src/evaluate_model.py
```

The evaluation report is saved to:

```text
reports/evaluation_report.json
```

### Current Evaluation Results

The latest locally generated evaluation report contains:

| Metric | Value |
|---|---:|
| Accuracy | 0.7559 |
| Precision | 0.5276 |
| Recall | 0.7674 |
| F1-score | 0.6253 |
| ROC-AUC | 0.8433 |

These values are from the current local evaluation report and may change when the model is retrained or the dataset changes.

## 🚦 Model Quality Gate

The quality gate checks whether the selected evaluation metric meets the configured minimum threshold.

Current configuration:

- **Metric:** Recall
- **Minimum threshold:** 0.50

Run the quality gate:

```bash
python src/quality_gate.py
```

The result is saved to:

```text
reports/quality_gate_report.json
```

The quality gate passes when the measured recall is greater than or equal to the configured minimum value.

## 📈 MLflow Experiment Tracking and Model Registry

MLflow is used to track model experiments, parameters, and evaluation metrics.

The project configuration specifies:

- **Experiment:** `Telco_Churn_Prediction`
- **Registered model:** `Telco_Churn_Production_Model`
- **Tracking URI:** `sqlite:///mlflow.db`

The registry lifecycle automation supports model stages such as Staging, Production, and Archived.

To view the MLflow interface locally, run:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

The MLflow database and generated model files are local project artifacts and may not be included in the Git repository.

## 🔁 Reproducibility

Reproducibility validation checks whether the pipeline produces consistent results under the configured conditions.

The project uses a fixed random state in its configuration:

```yaml
project:
  random_state: 42
```

This supports repeatable data splitting and model training when the same data, dependencies, and settings are used.

## 🤖 GitHub Actions – Continuous Integration

The project contains GitHub Actions workflows under:

```text
.github/workflows/
```

### Lab 8: Continuous Integration

Workflow file:

```text
lab8_ci.yml
```

The workflow is configured to run on selected pushes, pull requests, and manual dispatches. It checks configuration, runs model evaluation and the quality gate, and uploads reports.

### Lab 9: Failure Tests

Workflow file:

```text
lab9_failure_tests.yml
```

This workflow can be started manually through the **Actions** tab of the GitHub repository.

It runs evaluation, the quality gate, and failure-injection checks, then uploads available reports.

The failure-injection script checks missing dataset paths, missing model-artifact paths, invalid dataset schema, and missing processed-feature paths without deliberately deleting real project files.

*Note: Workflow files have been checked for YAML syntax locally. Successful execution on GitHub Actions must be verified in the repository's Actions tab.*

## 📄 Generated Reports

The main reports are stored in the `reports/` directory:

| File | Purpose |
|---|---|
| `evaluation_report.json` | Evaluation metrics and confusion matrix |
| `quality_gate_report.json` | Quality-gate decision and threshold |
| `lab9_failure_report.json` | Results of failure-injection checks |

Additional model registry reports may be generated under `artifacts/`.

## 👨‍💻 Author

**B. Surya Prakash**

CSE – Artificial Intelligence and Machine Learning (AIML)

**Project:** Telco Customer Churn Prediction – MLOps

# Disease Prediction using Hybrid Ensemble Model

An end-to-end machine learning pipeline that predicts diseases from patient symptoms using a **hybrid stacking ensemble**. The project combines multiple base classifiers with a meta-learner to deliver accurate, stable, and explainable multi-class disease diagnosis.

## Overview

Given a set of 132 binary symptom indicators (e.g. itching, joint pain, chills, skin rash), the model predicts the most likely disease (`prognosis`) out of 41 possible diagnoses — including conditions such as AIDS, Alcoholic Hepatitis, Allergy, Arthritis, Vertigo, and more.

The core idea is **stacked generalization**: instead of relying on a single algorithm, three diverse "expert" models each make predictions, and a meta-learner learns how to best combine their outputs.

## Model Architecture

**Level 0 — Base Experts**
| Model | Role |
|---|---|
| Random Forest (100 trees) | Captures non-linear feature interactions, robust to noise |
| SVM (RBF kernel, probability outputs) | Strong margin-based separation in high-dimensional symptom space |
| XGBoost | Gradient-boosted trees for high predictive power |

**Level 1 — Meta-Learner**
- Logistic Regression, trained on the out-of-fold predictions of the base models (5-fold internal CV via `StackingClassifier`)

## Pipeline

1. **Data cleaning** — drop empty columns (`Unnamed: 133`), audit class balance
2. **Preprocessing** — feature/target split, `StandardScaler` for feature scaling, `LabelEncoder` for the target
3. **Baseline evaluation** — Random Forest, SVM, and XGBoost trained and scored individually
4. **Hybrid ensemble** — `StackingClassifier` combining all three experts with a Logistic Regression meta-learner
5. **Evaluation** — accuracy, weighted recall, weighted F1, full classification report, and confusion matrix heatmap
6. **Stability analysis** — 5-fold cross-validation comparing baselines vs. the hybrid model across folds
7. **Explainability** — top-10 most influential symptoms extracted from the Random Forest expert's feature importances

## Results

| Model | Accuracy | Recall (weighted) | F1-Score (weighted) |
|---|---|---|---|
| Random Forest | 0.9762 | 0.9762 | 0.9762 |
| SVM | 0.9762 | 0.9762 | 0.9762 |
| XGBoost | 0.9762 | 0.9762 | 0.9762 |
| **Hybrid Ensemble** | **0.9762** | **0.9762** | **0.9762** |

The hybrid ensemble matches the strongest baselines on the held-out test set while showing more consistent accuracy across cross-validation folds, making it a more reliable choice for deployment.

## Repository Contents

| File | Description |
|---|---|
| `Disease_Prediction_ project_hybrid_enseamble.ipynb` | Full pipeline: preprocessing, training, evaluation, and visualizations |
| `.gitignore` | Ignores datasets, model artifacts, and environment files |

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Dataset

The notebook expects two CSV files loaded as `train_df` and `test_df`, each containing 132 symptom columns plus a `prognosis` target column (the standard Kaggle [Disease Prediction dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)).

### Run

```bash
jupyter notebook "Disease_Prediction_ project_hybrid_enseamble.ipynb"
```

Run the cells top to bottom to reproduce preprocessing, training, evaluation, and all plots.

## Visualizations Included

- Performance stability line plot (baselines vs. hybrid across CV folds)
- Confusion matrix heatmap for the hybrid model
- Top-10 symptom feature importance chart
- Per-model accuracy stability plots

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter

## Disclaimer

This project is for educational and research purposes only. It is **not** a substitute for professional medical diagnosis or advice.

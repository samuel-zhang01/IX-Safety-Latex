# Insurance Fraud Detection — Model Comparison Report

## Dataset Overview

- **Source:** `insurance fraud claims.csv`
- **Total samples:** 1000
- **Train / Test split:** 800 / 200 (80:20 stratified)
- **Features after preprocessing:** 76
- **Target:** `fraud_reported` — Fraud (Y) ≈ 24.7%, Not Fraud (N) ≈ 75.3%
- **Missing values:** `collision_type` (178), `property_damage` (360), `police_report_available` (343) — imputed with most-frequent value

## Preprocessing

| Step | Details |
|------|---------|
| Dropped columns | `_c39` (empty), `policy_number` (ID), `incident_location` (unique per row), `insured_zip` (near-unique) |
| Engineered features | `policy_age_days`, `incident_month`, `incident_day_of_week`, `csl_per_person`, `csl_per_accident` |
| Numerical scaling | `StandardScaler` |
| Low-cardinality categoricals (≤10 unique) | `OneHotEncoder` |
| Medium-cardinality categoricals (>10 unique) | `OrdinalEncoder` |
| Missing value imputation | Median (numeric), Mode (categorical) |

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| SVM (SVC) | 0.7550 | 0.0000 | 0.0000 | 0.0000 | 0.7638 |
|  **Logistic Regression** | 0.8100 | 0.6279 | 0.5510 | 0.5870 | 0.8069 |
| Decision Tree | 0.7400 | 0.4667 | 0.4286 | 0.4468 | 0.6348 |
| KNN | 0.7050 | 0.2917 | 0.1429 | 0.1918 | 0.5457 |

## Confusion Matrices

### SVM (SVC)

|  | Predicted: Not Fraud | Predicted: Fraud |
|--|---------------------|-----------------|
| **Actual: Not Fraud** | 151 | 0 |
| **Actual: Fraud** | 49 | 0 |

### Logistic Regression

|  | Predicted: Not Fraud | Predicted: Fraud |
|--|---------------------|-----------------|
| **Actual: Not Fraud** | 135 | 16 |
| **Actual: Fraud** | 22 | 27 |

### Decision Tree

|  | Predicted: Not Fraud | Predicted: Fraud |
|--|---------------------|-----------------|
| **Actual: Not Fraud** | 127 | 24 |
| **Actual: Fraud** | 28 | 21 |

### KNN

|  | Predicted: Not Fraud | Predicted: Fraud |
|--|---------------------|-----------------|
| **Actual: Not Fraud** | 134 | 17 |
| **Actual: Fraud** | 42 | 7 |

## Best Model Recommendation

**Logistic Regression** is the recommended model with an F1-score of **0.5870** and ROC-AUC of **0.8069**.

In fraud detection, recall (catching actual fraud cases) is critical because the cost of missing a fraudulent claim far exceeds the cost of investigating a false alarm. The F1-score balances precision and recall, making it the most appropriate primary metric for this imbalanced classification task.

> **Note:** Accuracy alone can be misleading on imbalanced datasets — a model predicting "Not Fraud" for every case would achieve ~75% accuracy but catch zero fraud.

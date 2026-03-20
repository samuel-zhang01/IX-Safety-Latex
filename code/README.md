# Supplementary Code

**Paper:** *Safe Learning to Defer with CVaR-Penalised Bayesian Uncertainty*

## Structure

```
code/
├── run_all_experiments.ipynb    Master notebook (run this)
├── fraud_model.py               Baseline classifiers (SVM, LR, DT, KNN)
├── fraud_model_advanced.py      + Bayesian, causal, XGBoost, Naive Bayes
├── data/
│   └── insurance_fraud_claims.csv
├── figures/                     Generated SVG figures (9 total)
└── reports/                     Generated Markdown reports
```

## Quick Start

```bash
cd code/
pip install numpy pandas scikit-learn xgboost matplotlib networkx scipy
jupyter notebook run_all_experiments.ipynb
```

Run all cells top-to-bottom. Everything regenerates from scratch.

## What Each File Does

| File | Purpose |
|------|---------|
| `fraud_model.py` | Loads data, engineers features, trains 4 baseline classifiers, outputs performance report |
| `fraud_model_advanced.py` | Adds XGBoost + Naive Bayes, runs Bayesian inference, McNemar's test, cross-validation, causal/propensity analysis, generates all 9 figures |
| `run_all_experiments.ipynb` | **Part A**: runs the full pipeline in one cell. **Part B**: one cell per figure for interactive inspection |

## Dataset

`data/insurance_fraud_claims.csv` — 1,000 insurance claims, 39 features, binary target `fraud_reported` (Y/N, ~25% fraud rate). Missing values in `collision_type`, `property_damage`, `police_report_available` are imputed automatically.

## Figures

| # | File | Shows |
|---|------|-------|
| 1 | `class_conditional_densities.svg` | P(feature | fraud) vs P(feature | not fraud) |
| 2 | `prior_posterior_update.svg` | Bayesian posterior update as features are observed |
| 3 | `roc_overlay.svg` | ROC curves for all 6 classifiers |
| 4 | `calibration_curves.svg` | Predicted probability vs actual fraud frequency |
| 5 | `cv_boxplot.svg` | 5-fold cross-validation F1 distributions |
| 6 | `mcnemar_heatmap.svg` | Pairwise McNemar's test p-values |
| 7 | `causal_dag.svg` | Hypothesised causal DAG for insurance fraud |
| 8 | `propensity_scores.svg` | Propensity score distributions + treatment effects |
| 9 | `feature_importance_comparison.svg` | Permutation importance vs KL divergence |

## Standalone Usage

```bash
python fraud_model.py            # -> reports/fraud_model_report.md
python fraud_model_advanced.py   # -> reports/ + figures/
```

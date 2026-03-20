"""
Insurance Fraud Detection — Model Comparison Pipeline
Trains SVM, Logistic Regression, Decision Tree, and KNN classifiers
on insurance claims data and generates a Markdown performance report.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
import os
import warnings

warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, "data", "insurance_fraud_claims.csv")
REPORT_PATH = os.path.join(_DIR, "reports", "fraud_model_report.md")
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ── Step 1: Load & Clean ────────────────────────────────────────────────────

def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Drop empty trailing column and identifier
    if "_c39" in df.columns:
        df.drop(columns=["_c39"], inplace=True)
    df.drop(columns=["policy_number"], inplace=True)

    # Replace "?" placeholders with NaN
    df.replace("?", np.nan, inplace=True)

    print(f"  Shape: {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"  Missing values:\n{missing.to_string(header=False)}")
    return df


# ── Step 2: Feature Engineering ──────────────────────────────────────────────

def engineer_features(df):
    # Encode target
    df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # Date features
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])

    df["policy_age_days"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
    df["incident_month"] = df["incident_date"].dt.month
    df["incident_day_of_week"] = df["incident_date"].dt.dayofweek

    df.drop(columns=["incident_date", "policy_bind_date"], inplace=True)

    # Split compound policy_csl (e.g. "100/300") into two numeric columns
    csl_split = df["policy_csl"].str.split("/", expand=True).astype(int)
    df["csl_per_person"] = csl_split[0]
    df["csl_per_accident"] = csl_split[1]
    df.drop(columns=["policy_csl"], inplace=True)

    # Drop near-unique columns with no predictive value
    df.drop(columns=["incident_location", "insured_zip"], inplace=True)

    print(f"  Shape after engineering: {df.shape}")
    return df


# ── Step 3: Column Classification ───────────────────────────────────────────

def classify_columns(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    low_cat_cols = [c for c in cat_cols if X[c].nunique() <= 10]
    med_cat_cols = [c for c in cat_cols if X[c].nunique() > 10]

    return X, y, num_cols, low_cat_cols, med_cat_cols


# ── Step 4: Preprocessor ────────────────────────────────────────────────────

def build_preprocessor(num_cols, low_cat_cols, med_cat_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    low_cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    med_cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("low_cat", low_cat_transformer, low_cat_cols),
            ("med_cat", med_cat_transformer, med_cat_cols),
        ],
        remainder="drop",
    )


# ── Step 5: Models ──────────────────────────────────────────────────────────

def get_models(preprocessor):
    return {
        "SVM (SVC)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


# ── Step 6: Evaluation ──────────────────────────────────────────────────────

def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    print(f"\n  {'='*46}")
    print(f"  {name}")
    print(f"  {'='*46}")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1-Score:  {result['f1']:.4f}")
    print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
    cm = result["confusion_matrix"]
    print(f"  Confusion Matrix: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    return result


# ── Step 7: Report Generation ────────────────────────────────────────────────

def generate_report(results, train_size, test_size, feature_count, filepath):
    best = max(results, key=lambda r: r["f1"])

    lines = []
    lines.append("# Insurance Fraud Detection — Model Comparison Report\n")

    # Dataset overview
    lines.append("## Dataset Overview\n")
    lines.append(f"- **Source:** `insurance fraud claims.csv`")
    lines.append(f"- **Total samples:** {train_size + test_size}")
    lines.append(f"- **Train / Test split:** {train_size} / {test_size} (80:20 stratified)")
    lines.append(f"- **Features after preprocessing:** {feature_count}")
    lines.append("- **Target:** `fraud_reported` — Fraud (Y) ≈ 24.7%, Not Fraud (N) ≈ 75.3%")
    lines.append("- **Missing values:** `collision_type` (178), `property_damage` (360), "
                  "`police_report_available` (343) — imputed with most-frequent value\n")

    # Preprocessing summary
    lines.append("## Preprocessing\n")
    lines.append("| Step | Details |")
    lines.append("|------|---------|")
    lines.append("| Dropped columns | `_c39` (empty), `policy_number` (ID), "
                  "`incident_location` (unique per row), `insured_zip` (near-unique) |")
    lines.append("| Engineered features | `policy_age_days`, `incident_month`, "
                  "`incident_day_of_week`, `csl_per_person`, `csl_per_accident` |")
    lines.append("| Numerical scaling | `StandardScaler` |")
    lines.append("| Low-cardinality categoricals (≤10 unique) | `OneHotEncoder` |")
    lines.append("| Medium-cardinality categoricals (>10 unique) | `OrdinalEncoder` |")
    lines.append("| Missing value imputation | Median (numeric), Mode (categorical) |\n")

    # Performance table
    lines.append("## Model Performance Comparison\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
    lines.append("|-------|----------|-----------|--------|----------|---------|")
    for r in results:
        marker = " **" if r["name"] == best["name"] else ""
        end = "**" if marker else ""
        lines.append(
            f"| {marker}{r['name']}{end} "
            f"| {r['accuracy']:.4f} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['roc_auc']:.4f} |"
        )
    lines.append("")

    # Confusion matrices
    lines.append("## Confusion Matrices\n")
    for r in results:
        cm = r["confusion_matrix"]
        lines.append(f"### {r['name']}\n")
        lines.append("|  | Predicted: Not Fraud | Predicted: Fraud |")
        lines.append("|--|---------------------|-----------------|")
        lines.append(f"| **Actual: Not Fraud** | {cm[0][0]} | {cm[0][1]} |")
        lines.append(f"| **Actual: Fraud** | {cm[1][0]} | {cm[1][1]} |\n")

    # Recommendation
    lines.append("## Best Model Recommendation\n")
    lines.append(f"**{best['name']}** is the recommended model with an "
                  f"F1-score of **{best['f1']:.4f}** and ROC-AUC of **{best['roc_auc']:.4f}**.\n")
    lines.append("In fraud detection, recall (catching actual fraud cases) is critical because "
                  "the cost of missing a fraudulent claim far exceeds the cost of investigating "
                  "a false alarm. The F1-score balances precision and recall, making it the most "
                  "appropriate primary metric for this imbalanced classification task.\n")
    lines.append("> **Note:** Accuracy alone can be misleading on imbalanced datasets — a model "
                  "predicting \"Not Fraud\" for every case would achieve ~75% accuracy but "
                  "catch zero fraud.\n")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Insurance Fraud Detection — Model Training Pipeline")
    print("=" * 60)

    print("\n[1/7] Loading and cleaning data...")
    df = load_and_clean(CSV_PATH)

    print("\n[2/7] Engineering features...")
    df = engineer_features(df)

    print("\n[3/7] Classifying columns...")
    X, y, num_cols, low_cat_cols, med_cat_cols = classify_columns(df, "fraud_reported")
    print(f"  Numerical:              {len(num_cols)}")
    print(f"  Low-card categoricals:  {len(low_cat_cols)}")
    print(f"  Med-card categoricals:  {len(med_cat_cols)}")

    print("\n[4/7] Building preprocessing pipeline...")
    preprocessor = build_preprocessor(num_cols, low_cat_cols, med_cat_cols)

    print("\n[5/7] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Train fraud rate: {y_train.mean():.3f}")
    print(f"  Test fraud rate:  {y_test.mean():.3f}")

    print("\n[6/7] Training and evaluating models...")
    models = get_models(preprocessor)
    all_results = []

    for name, pipeline in models.items():
        print(f"\n  Training {name}...")
        pipeline.fit(X_train, y_train)
        result = evaluate_model(name, pipeline, X_test, y_test)
        all_results.append(result)

    # Determine feature count from the fitted preprocessor
    sample_transformed = models[list(models.keys())[0]].named_steps["preprocessor"].transform(
        X_test.head(1)
    )
    feature_count = sample_transformed.shape[1]

    print("\n[7/7] Generating markdown report...")
    generate_report(all_results, X_train.shape[0], X_test.shape[0], feature_count, REPORT_PATH)
    print(f"  Report saved to: {REPORT_PATH}")

    best = max(all_results, key=lambda r: r["f1"])
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best['name']} (F1={best['f1']:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

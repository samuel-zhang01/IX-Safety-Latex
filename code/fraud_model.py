"""
Insurance Fraud Detection — Base Utilities
Provides data loading, feature engineering, column classification,
and preprocessing pipeline construction.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_clean(csv_path):
    """Load the insurance fraud CSV, clean missing values and encode target."""
    df = pd.read_csv(csv_path)

    # Drop the trailing empty column if present
    if "_c39" in df.columns:
        df = df.drop(columns=["_c39"])

    # Map target to binary
    df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Impute categorical columns that have missing values
    for col in ["collision_type", "property_damage", "police_report_available"]:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

    # Parse date columns
    for col in ["policy_bind_date", "incident_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def engineer_features(df):
    """Add derived features useful for fraud detection."""
    df = df.copy()

    # Days between policy bind and incident
    if "policy_bind_date" in df.columns and "incident_date" in df.columns:
        df["policy_age_days"] = (df["incident_date"] - df["policy_bind_date"]).dt.days

    # Claim-to-premium ratio
    if "total_claim_amount" in df.columns and "policy_annual_premium" in df.columns:
        df["claim_to_premium_ratio"] = df["total_claim_amount"] / (
            df["policy_annual_premium"] + 1e-6
        )

    # Injury-to-vehicle claim ratio
    if "injury_claim" in df.columns and "vehicle_claim" in df.columns:
        df["injury_to_vehicle_ratio"] = df["injury_claim"] / (
            df["vehicle_claim"] + 1
        )

    return df


def classify_columns(df, target_col):
    """
    Separate features into numeric, low-cardinality categorical, and
    medium-cardinality categorical. Returns X, y, and the three column lists.
    """
    y = df[target_col].copy()

    # Drop target, date columns, and identifiers
    drop_cols = [target_col]
    for col in df.columns:
        if df[col].dtype in ["datetime64[ns]", "<M8[ns]"]:
            drop_cols.append(col)
    if "policy_number" in df.columns:
        drop_cols.append("policy_number")
    if "insured_zip" in df.columns:
        drop_cols.append("insured_zip")
    if "incident_location" in df.columns:
        drop_cols.append("incident_location")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    low_cat_cols = [c for c in cat_cols if X[c].nunique() <= 5]
    med_cat_cols = [c for c in cat_cols if 5 < X[c].nunique() <= 50]

    # High cardinality categorical columns (>50) are dropped
    high_cat_cols = [c for c in cat_cols if X[c].nunique() > 50]
    if high_cat_cols:
        X = X.drop(columns=high_cat_cols)

    return X, y, num_cols, low_cat_cols, med_cat_cols


def build_preprocessor(num_cols, low_cat_cols, med_cat_cols):
    """Build a ColumnTransformer with appropriate transformations."""
    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if low_cat_cols:
        cat_low_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat_low", cat_low_pipeline, low_cat_cols))

    if med_cat_cols:
        cat_med_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ])
        transformers.append(("cat_med", cat_med_pipeline, med_cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")

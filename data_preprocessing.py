"""
data_preprocessing.py - Data Loading, Cleaning, and Feature Engineering
========================================================================
Unit 1 – Introduction to Responsible AI

This module handles:
  • Loading the IBM HR Analytics dataset
  • Cleaning and encoding categorical variables
  • Feature engineering (AgeGroup, IncomeGroup)
  • Identifying sensitive attributes (Gender, Age)
  • Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import DATASET_PATH, log


def load_dataset():
    """Load the IBM HR Employee Attrition dataset."""
    log("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    log(f"Dataset shape: {df.shape}")
    return df


def clean_data(df):
    """
    Clean the dataset:
      - Drop constant / non-informative columns
      - Handle missing values (none expected, but defensive)
      - Map the target variable (Attrition) to binary 0/1
    """
    log("Cleaning data...")

    # Drop columns that carry no predictive value
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Map target: 'Yes' → 1 (left / not hired analogy), 'No' → 0
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Fill any rare missing values
    df = df.fillna(df.median(numeric_only=True))

    log(f"After cleaning: {df.shape}")
    return df


def engineer_features(df):
    """
    Create additional features for analysis:
      - AgeGroup: bins employees into age brackets
      - IncomeGroup: quartile-based income categories
      - Gender_Binary: numeric gender encoding
    """
    log("Engineering features...")

    # Age groups (sensitive attribute - used for fairness analysis)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 30, 40, 50, 61],
        labels=["18-30", "31-40", "41-50", "51-60"],
    )

    # Income groups
    df["IncomeGroup"] = pd.qcut(
        df["MonthlyIncome"], q=4, labels=["Low", "Medium", "High", "Very High"]
    )

    # Binary gender encoding (Female=0, Male=1)
    df["Gender_Binary"] = (df["Gender"] == "Male").astype(int)

    return df


def encode_categoricals(df):
    """Label-encode all remaining categorical string columns."""
    log("Encoding categorical variables...")
    le = LabelEncoder()
    encoders = {}
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def prepare_data(df, target="Attrition", test_size=0.2, random_state=42):
    """
    Prepare features (X) and target (y), apply scaling, return train/test split.
    Also returns sensitive attribute arrays for fairness analysis.
    """
    log("Preparing train/test split...")

    # Store sensitive attributes BEFORE dropping them from features
    sensitive = {
        "Gender": df["Gender_Binary"].values,
        "AgeGroup": df["AgeGroup"].values if "AgeGroup" in df.columns else None,
    }

    # Features: drop target and helper columns
    exclude = [target, "AgeGroup", "IncomeGroup"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target].values

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Align sensitive attribute indices
    train_idx = X_train.index
    test_idx = X_test.index
    sensitive_train = {k: v[train_idx] for k, v in sensitive.items() if v is not None}
    sensitive_test = {k: v[test_idx] for k, v in sensitive.items() if v is not None}

    log(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    log(f"Target distribution (train): {np.bincount(y_train)}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "sensitive_train": sensitive_train,
        "sensitive_test": sensitive_test,
        "scaler": scaler,
    }


def run_preprocessing():
    """Full preprocessing pipeline – returns cleaned data + split."""
    df = load_dataset()
    df = clean_data(df)
    df = engineer_features(df)
    df_encoded, encoders = encode_categoricals(df)
    data = prepare_data(df_encoded)
    data["dataframe"] = df_encoded
    data["encoders"] = encoders
    return data


if __name__ == "__main__":
    data = run_preprocessing()
    print("\nFeature columns:", data["feature_names"])
    print("Training samples:", data["X_train"].shape[0])
    print("Test samples:", data["X_test"].shape[0])

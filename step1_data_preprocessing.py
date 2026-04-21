"""
Step 1: Data Preprocessing for AQI Prediction System
=====================================================
Loads city_day.csv, cleans it, handles missing values,
detects outliers, scales features and saves everything
needed for model training (Step 2) and inference (Step 3).

Run:
    python step1_data_preprocessing.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    DATASET_PATH, OUTPUT_DIR,
    FEATURE_NAMES, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE,
    SCALER_PATH, IMPUTER_PATH,
    X_TRAIN_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH,
    DATA_STATS_PATH, PREPROC_RPT_PATH,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place city_day.csv in the project root."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns we need."""
    required = FEATURE_NAMES + [TARGET_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    return df[required].copy()


def remove_missing_target(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where AQI (target) is missing."""
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    dropped = before - len(df)
    print(f"[INFO] Dropped {dropped} rows with missing AQI")
    return df, dropped


def impute_features(df: pd.DataFrame, imputer=None):
    """Impute missing feature values with column median."""
    X = df[FEATURE_NAMES]
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        print("[INFO] Fitted new median imputer")
    else:
        X_imputed = imputer.transform(X)
        print("[INFO] Applied existing imputer")

    df[FEATURE_NAMES] = X_imputed
    return df, imputer


def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 3.0) -> tuple[pd.DataFrame, int]:
    """
    Remove outlier rows using the IQR method.

    A row is considered an outlier if any column value falls outside
    [Q1 - multiplier * IQR, Q3 + multiplier * IQR].
    A generous multiplier (default 3×) retains most legitimate data
    while removing extreme anomalies.
    """
    before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)

    for col in FEATURE_NAMES + [TARGET_COLUMN]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        col_mask = df[col].between(lower, upper)
        mask = mask & col_mask

    df_clean = df[mask]
    dropped = before - len(df_clean)
    print(f"[INFO] Removed {dropped} outlier rows ({dropped / before * 100:.1f}%)")
    return df_clean, dropped


def compute_data_stats(df: pd.DataFrame) -> dict:
    """Compute min/max/mean/median/std for each feature."""
    stats = {}
    for col in FEATURE_NAMES:
        stats[col] = {
            "min":    round(float(df[col].min()),    4),
            "max":    round(float(df[col].max()),    4),
            "mean":   round(float(df[col].mean()),   4),
            "median": round(float(df[col].median()), 4),
            "std":    round(float(df[col].std()),    4),
        }
    stats[TARGET_COLUMN] = {
        "min":    round(float(df[TARGET_COLUMN].min()),    4),
        "max":    round(float(df[TARGET_COLUMN].max()),    4),
        "mean":   round(float(df[TARGET_COLUMN].mean()),   4),
        "median": round(float(df[TARGET_COLUMN].median()), 4),
        "std":    round(float(df[TARGET_COLUMN].std()),    4),
    }
    return stats


def scale_features(X_train: np.ndarray, X_test: np.ndarray, scaler=None):
    """Standardize features to zero mean and unit variance."""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("[INFO] Fitted new StandardScaler")
    else:
        X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def preprocess_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = ["=" * 60, "AQI PREDICTION — DATA PREPROCESSING REPORT", "=" * 60, ""]

    # 1. Load
    df_raw = load_data(DATASET_PATH)
    report_lines.append(f"Raw dataset shape: {df_raw.shape}")

    # 2. Select features + target
    df = select_features(df_raw)
    report_lines.append(f"Selected columns: {list(df.columns)}")

    # 3. Remove rows with missing AQI
    df, n_missing_target = remove_missing_target(df)
    report_lines.append(f"Rows dropped (missing AQI): {n_missing_target}")

    # 4. Impute missing feature values
    df, imputer = impute_features(df)
    n_imputed = df[FEATURE_NAMES].isna().sum().sum()
    report_lines.append(f"Remaining NaNs in features after imputation: {n_imputed}")

    # 5. Remove outliers
    df_clean, n_outliers = remove_outliers_iqr(df)
    report_lines.append(f"Rows removed as outliers: {n_outliers}")
    report_lines.append(f"Clean dataset shape: {df_clean.shape}")

    # 6. Compute stats (before scaling, for reference)
    stats = compute_data_stats(df_clean)
    report_lines.append("\nFeature statistics (after cleaning, before scaling):")
    for col, s in stats.items():
        report_lines.append(
            f"  {col:<8}  min={s['min']:.2f}  max={s['max']:.2f}  "
            f"mean={s['mean']:.2f}  std={s['std']:.2f}"
        )

    # 7. Split
    X = df_clean[FEATURE_NAMES].values
    y = df_clean[TARGET_COLUMN].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    report_lines.append(f"\nTrain samples : {len(X_train_raw)}")
    report_lines.append(f"Test  samples : {len(X_test_raw)}")

    # 8. Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_raw, X_test_raw)

    # 9. Save everything
    joblib.dump(imputer,       IMPUTER_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(X_train_scaled, X_TRAIN_PATH)
    joblib.dump(X_test_scaled,  X_TEST_PATH)
    joblib.dump(y_train,        Y_TRAIN_PATH)
    joblib.dump(y_test,         Y_TEST_PATH)

    with open(DATA_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    report_lines.append("\nSaved artifacts:")
    for p in [IMPUTER_PATH, SCALER_PATH, X_TRAIN_PATH, X_TEST_PATH,
              Y_TRAIN_PATH, Y_TEST_PATH, DATA_STATS_PATH]:
        report_lines.append(f"  ✅ {p}")

    report_lines += ["", "=" * 60, "Preprocessing complete!", "=" * 60]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(PREPROC_RPT_PATH, "w") as f:
        f.write(report_text)

    return {
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
        "imputer": imputer,
        "stats":   stats,
    }


if __name__ == "__main__":
    preprocess_data()

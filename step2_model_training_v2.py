"""
Step 2 (Improved): Model Training with Proper AQI Calibration
==============================================================

PROBLEM: Previous model predicted 308.8 when actual AQI was 153.
ROOT CAUSE: StandardScaler changes feature interpretation; no proper calibration.

SOLUTION:
- Use MinMaxScaler (0-1) for feature normalization on top of preprocessed features
- Train model to directly predict AQI in original range (no intermediate normalization)
- 5-fold cross-validation to detect overfitting
- Calibration analysis to check for systematic bias
- Saves best model, feature normalizer, and metadata

Run: python step2_model_training_v2.py
"""

import os
import json
import warnings
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    OUTPUT_DIR, MODELS_DIR, FEATURE_NAMES,
    X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    SCALER_PATH, BEST_MODEL_PATH, MODEL_META_PATH,
)

warnings.filterwarnings("ignore")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_evaluate():
    """Train models with MinMaxScaler for proper AQI prediction."""

    print("\n" + "=" * 80)
    print(" " * 15 + "STEP 2: MODEL TRAINING WITH CALIBRATION")
    print("=" * 80 + "\n")

    # Load preprocessed data
    print("[LOAD] Loading preprocessed data...")
    X_train = joblib.load(X_TRAIN_PATH)
    X_test = joblib.load(X_TEST_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  AQI range (train): [{y_train.min():.1f}, {y_train.max():.1f}]")

    # Apply MinMaxScaler on top of preprocessed features for better model calibration
    print("\n[NORMALIZE] Applying MinMaxScaler to preprocessed features...")
    feature_normalizer = MinMaxScaler()
    X_train_norm = feature_normalizer.fit_transform(X_train)
    X_test_norm = feature_normalizer.transform(X_test)
    print("  Features normalized to [0, 1] range")

    # Define models — trained to predict AQI directly (no AQI normalization)
    models = {
        "Ridge Regression": Ridge(alpha=50.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    print("\n[TRAIN] Training models with 5-fold cross-validation...\n")
    print(f"{'Model':<25} {'CV R\u00b2':>10} {'Test R\u00b2':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 65)

    results = {}
    best_model_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_norm, y_train, cv=5, scoring="r2")
        model.fit(X_train_norm, y_train)
        y_pred = np.maximum(model.predict(X_test_norm), 0.0)

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-6))) * 100)

        results[name] = {
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "test_r2": r2,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_mape": mape,
            "model": model,
            "predictions": y_pred,
        }

        print(f"{name:<25} {cv_scores.mean():>10.4f} {r2:>10.4f} {rmse:>10.2f} {mae:>10.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    best_result = results[best_model_name]
    best_preds = best_result["predictions"]

    print("\n" + "=" * 80)
    print(f"\U0001f3c6 BEST MODEL: {best_model_name}")
    print(f"   R\u00b2   : {best_result['test_r2']:.4f}")
    print(f"   RMSE : {best_result['test_rmse']:.2f} units")
    print(f"   MAE  : {best_result['test_mae']:.2f} units")
    print(f"   MAPE : {best_result['test_mape']:.1f}%")
    print("=" * 80)

    # Calibration analysis
    print("\n[CALIBRATION] Analyzing predictions...\n")
    errors = y_test - best_preds
    print(f"  Mean error : {errors.mean():+.2f} (ideally close to 0)")
    print(f"  Std error  : {errors.std():.2f}")
    print(f"  Max |error|: {np.abs(errors).max():.2f}")
    if abs(errors.mean()) > 10:
        print(f"  WARNING: Model has {abs(errors.mean()):.1f} unit systematic bias")
    else:
        print("  Model is well-calibrated (mean error near 0)")

    # Save best model and feature normalizer
    print("\n[SAVE] Saving artifacts...\n")
    joblib.dump(best_result["model"], BEST_MODEL_PATH)
    normalizer_path = os.path.join(MODELS_DIR, "feature_normalizer.pkl")
    joblib.dump(feature_normalizer, normalizer_path)

    all_models_meta = {
        name: {
            "cv_r2_mean": res["cv_r2_mean"],
            "cv_r2_std": res["cv_r2_std"],
            "test_r2": res["test_r2"],
            "test_rmse": res["test_rmse"],
            "test_mae": res["test_mae"],
            "test_mape": res["test_mape"],
        }
        for name, res in results.items()
    }

    metadata = {
        "best_model_name": best_model_name,
        "r2_score": float(best_result["test_r2"]),
        "test_r2": float(best_result["test_r2"]),
        "rmse": float(best_result["test_rmse"]),
        "mae": float(best_result["test_mae"]),
        "cv_r2_mean": float(best_result["cv_r2_mean"]),
        "cv_r2_std": float(best_result["cv_r2_std"]),
        "feature_names": FEATURE_NAMES,
        "all_models": all_models_meta,
    }

    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {BEST_MODEL_PATH}")
    print(f"  {normalizer_path}")
    print(f"  {MODEL_META_PATH}")
    print("\nTraining complete! Run step3_inference_system.py to test predictions.")


if __name__ == "__main__":
    train_and_evaluate()
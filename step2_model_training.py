"""
Step 2: Model Training for AQI Prediction System
=================================================
Loads preprocessed data from Step 1, trains three regression
models with 5-fold cross-validation, selects the best model
by R² score, and saves it together with metrics and diagnostics.

Run:
    python step2_model_training.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    OUTPUT_DIR, MODELS_DIR,
    FEATURE_NAMES,
    X_TRAIN_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH,
    BEST_MODEL_PATH, MODEL_META_PATH,
    FEAT_IMP_PATH, TRAINING_RPT_PATH,
    CV_FOLDS, RANDOM_STATE,
    RF_PARAMS, GB_PARAMS,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Zero-valued true labels are excluded to avoid division by zero.
    Note: if many true values are zero or near-zero the reported MAPE
    may under-represent the actual error on those samples.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dict of regression metrics on the test set."""
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    mape_val = mape(y_test, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape_val}


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Return mean ± std of R² across CV folds."""
    scores = cross_val_score(
        model, X_train, y_train,
        cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    return {"mean_r2": float(scores.mean()), "std_r2": float(scores.std())}


def get_feature_importance(model, model_name: str) -> dict | None:
    """Extract feature importance for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(FEATURE_NAMES, importances.tolist()))
    return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load preprocessed data ──────────────────
    for path in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run step1_data_preprocessing.py first."
            )

    X_train = joblib.load(X_TRAIN_PATH)
    X_test  = joblib.load(X_TEST_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    y_test  = joblib.load(Y_TEST_PATH)

    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set    : {X_test.shape[0]} samples")

    # ── Define models ────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(**RF_PARAMS),
        "Gradient Boosting": GradientBoostingRegressor(**GB_PARAMS),
    }

    report_lines = ["=" * 65, "AQI PREDICTION — MODEL TRAINING REPORT", "=" * 65, ""]
    all_results  = {}
    best_name    = None
    best_r2      = -np.inf
    best_model   = None

    for name, model in models.items():
        print(f"\n[INFO] Training: {name} …")

        # Cross-validation
        cv_res = cross_validate_model(model, X_train, y_train)
        print(f"       CV R²: {cv_res['mean_r2']:.4f} ± {cv_res['std_r2']:.4f}")

        # Full train on training set
        model.fit(X_train, y_train)

        # Test-set evaluation
        metrics = evaluate_model(model, X_test, y_test)
        print(
            f"       Test  R²={metrics['R2']:.4f}  "
            f"RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  "
            f"MAPE={metrics['MAPE']:.2f}%"
        )

        feat_imp = get_feature_importance(model, name)

        all_results[name] = {
            "cv_r2_mean":  cv_res["mean_r2"],
            "cv_r2_std":   cv_res["std_r2"],
            "test_r2":     metrics["R2"],
            "test_rmse":   metrics["RMSE"],
            "test_mae":    metrics["MAE"],
            "test_mape":   metrics["MAPE"],
            "feature_importance": feat_imp,
        }

        report_lines.append(f"Model: {name}")
        report_lines.append(
            f"  CV R²    : {cv_res['mean_r2']:.4f} ± {cv_res['std_r2']:.4f}"
        )
        report_lines.append(f"  Test R²  : {metrics['R2']:.4f}")
        report_lines.append(f"  Test RMSE: {metrics['RMSE']:.2f}")
        report_lines.append(f"  Test MAE : {metrics['MAE']:.2f}")
        report_lines.append(f"  Test MAPE: {metrics['MAPE']:.2f}%")
        if feat_imp:
            sorted_fi = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
            report_lines.append(
                "  Feature Importance: "
                + ", ".join(f"{k}={v:.3f}" for k, v in sorted_fi)
            )
        report_lines.append("")

        if metrics["R2"] > best_r2:
            best_r2    = metrics["R2"]
            best_name  = name
            best_model = model

    # ── Save best model ─────────────────────────
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n[INFO] Best model: {best_name}  (R²={best_r2:.4f})")
    print(f"[INFO] Saved to: {BEST_MODEL_PATH}")

    # ── Model metadata ───────────────────────────
    best_res = all_results[best_name]
    metadata = {
        "best_model_name": best_name,
        "features":        FEATURE_NAMES,
        "r2_score":        best_res["test_r2"],
        "rmse":            best_res["test_rmse"],
        "mae":             best_res["test_mae"],
        "mape":            best_res["test_mape"],
        "cv_r2_mean":      best_res["cv_r2_mean"],
        "cv_r2_std":       best_res["cv_r2_std"],
        "cv_folds":        CV_FOLDS,
        "all_models":      all_results,
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Feature importance CSV ───────────────────
    best_fi = best_res.get("feature_importance")
    if best_fi:
        fi_df = (
            pd.DataFrame(list(best_fi.items()), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.to_csv(FEAT_IMP_PATH, index=False)
        report_lines.append(f"Feature Importance ({best_name}):")
        for _, row in fi_df.iterrows():
            report_lines.append(f"  {row['Feature']:<8} {row['Importance']:.4f}")
        report_lines.append("")

    # ── Training report ──────────────────────────
    report_lines += [
        f"Best Model : {best_name}",
        f"Best R²    : {best_r2:.4f}",
        "",
        "=" * 65,
        "Training complete!",
        "=" * 65,
    ]

    # Residual statistics for best model
    y_pred_best = best_model.predict(X_test)
    residuals   = y_test - y_pred_best
    report_lines += [
        "",
        "Residual Analysis (best model on test set):",
        f"  Mean residual    : {residuals.mean():.4f}",
        f"  Std  residual    : {residuals.std():.4f}",
        f"  Max  residual    : {residuals.max():.4f}",
        f"  Min  residual    : {residuals.min():.4f}",
        f"  % within ±10 AQI: "
        f"{(np.abs(residuals) <= 10).mean() * 100:.1f}%",
        f"  % within ±20 AQI: "
        f"{(np.abs(residuals) <= 20).mean() * 100:.1f}%",
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(TRAINING_RPT_PATH, "w") as f:
        f.write(report_text)

    return best_model, metadata


if __name__ == "__main__":
    train_models()

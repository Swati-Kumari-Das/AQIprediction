"""
Step 3: Inference System for AQI Prediction
============================================
Provides the AQIPredictor class that wraps the saved model
and handles input validation, imputation, scaling, prediction,
AQI categorisation, and health advice.

Usage (programmatic):
    from step3_inference_system import AQIPredictor

    predictor = AQIPredictor()
    result = predictor.predict({
        'PM2.5': 65, 'PM10': 120, 'NO2': 45,
        'SO2': 20, 'CO': 1.5, 'O3': 35
    })
    print(result)

Usage (CLI):
    python step3_inference_system.py
"""

import os
import json
import warnings
import numpy as np
import joblib
import requests

from config import (
    FEATURE_NAMES, FEATURE_RANGES, AQI_CATEGORIES, HEALTH_ADVICE,
    BEST_MODEL_PATH, SCALER_PATH, IMPUTER_PATH,
    MODEL_META_PATH, DATA_STATS_PATH,
    AQICN_API_TOKEN, AQICN_API_URL,
)

warnings.filterwarnings("ignore")


class AQIPredictor:
    """
    Encapsulates loading the trained model + preprocessing objects
    and exposes simple prediction / categorisation methods.
    """

    def __init__(self):
        self._model   = None
        self._scaler  = None
        self._imputer = None
        self._metadata = {}
        self._stats    = {}
        self._load_artifacts()

    # ─────────────────────────────────────────────
    # Artifact loading
    # ─────────────────────────────────────────────

    def _load_artifacts(self):
        """Load model, scaler, imputer and metadata from disk."""
        missing = [
            p for p in [BEST_MODEL_PATH, SCALER_PATH, IMPUTER_PATH]
            if not os.path.exists(p)
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing trained artifacts: {missing}\n"
                "Run step1_data_preprocessing.py and step2_model_training.py first."
            )

        self._model = joblib.load(BEST_MODEL_PATH)
        self._scaler = joblib.load(SCALER_PATH)
        self._imputer = joblib.load(IMPUTER_PATH)

        if os.path.exists(MODEL_META_PATH):
            with open(MODEL_META_PATH) as f:
                self._metadata = json.load(f)

        if os.path.exists(DATA_STATS_PATH):
            with open(DATA_STATS_PATH) as f:
                self._stats = json.load(f)

    # ─────────────────────────────────────────────
    # Input validation
    # ─────────────────────────────────────────────

    def validate_input(self, pollutant_dict: dict) -> tuple[bool, str]:
        """
        Validate a dict of pollutant values.

        Returns:
            (True, "")               if valid
            (False, error_message)   if invalid
        """
        errors = []
        for feature in FEATURE_NAMES:
            val = pollutant_dict.get(feature)
            if val is None:
                continue  # missing values are imputed later
            try:
                val = float(val)
            except (TypeError, ValueError):
                errors.append(f"{feature}: value '{val}' is not a number.")
                continue

            rng = FEATURE_RANGES.get(feature, {})
            lo  = rng.get("min", 0)
            hi  = rng.get("max", float("inf"))
            if not (lo <= val <= hi):
                errors.append(
                    f"{feature}: {val} is outside valid range "
                    f"[{lo}, {hi}] {rng.get('unit', '')}."
                )

        if errors:
            return False, " | ".join(errors)
        return True, ""

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────

    def predict(self, pollutant_dict: dict) -> dict:
        """
        Predict the current AQI from pollutant values.

        Parameters
        ----------
        pollutant_dict : dict
            Keys should be a subset of FEATURE_NAMES.
            Missing keys are imputed.

        Returns
        -------
        dict with keys:
            AQI           : float
            Category      : str
            Color         : str  (hex)
            Emoji         : str
            Description   : str
            Health_Advice : list[str]
            Input_Values  : dict
            Confidence    : str  (High / Medium / Low)
        """
        # Validate
        valid, err = self.validate_input(pollutant_dict)
        if not valid:
            raise ValueError(f"Input validation failed: {err}")

        # Build feature vector (NaN for missing features)
        row = np.array(
            [float(pollutant_dict.get(f, np.nan)) for f in FEATURE_NAMES],
            dtype=float,
        ).reshape(1, -1)

        # Impute then scale
        row_imputed = self._imputer.transform(row)
        row_scaled  = self._scaler.transform(row_imputed)

        # Predict
        aqi_raw = float(self._model.predict(row_scaled)[0])
        aqi     = max(0.0, round(aqi_raw, 1))

        # Categorise
        cat = self.get_category(aqi)

        # Confidence: based on how many features were provided
        n_provided = sum(1 for f in FEATURE_NAMES if pollutant_dict.get(f) is not None)
        if n_provided >= 5:
            confidence = "High"
        elif n_provided >= 3:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "AQI":          aqi,
            "Category":     cat["name"],
            "Color":        cat["color"],
            "Emoji":        cat["emoji"],
            "Description":  cat["description"],
            "Health_Advice": self.get_health_advice(aqi),
            "Input_Values": {
                f: float(pollutant_dict.get(f, np.nan))
                for f in FEATURE_NAMES
            },
            "Confidence": confidence,
        }

    # ─────────────────────────────────────────────
    # AQI categorisation
    # ─────────────────────────────────────────────

    def get_category(self, aqi: float) -> dict:
        """Return the AQI category dict for a given AQI value."""
        for cat in AQI_CATEGORIES:
            lo, hi = cat["range"]
            if lo <= aqi <= hi:
                return cat
        # Above the last threshold
        return AQI_CATEGORIES[-1]

    # ─────────────────────────────────────────────
    # Health advice
    # ─────────────────────────────────────────────

    def get_health_advice(self, aqi: float) -> list[str]:
        """Return a list of health suggestions for a given AQI."""
        cat = self.get_category(aqi)
        return HEALTH_ADVICE.get(cat["name"], ["No advice available."])

    # ─────────────────────────────────────────────
    # Live API data fetch
    # ─────────────────────────────────────────────

    def fetch_api_data(self, city: str) -> dict:
        """
        Fetch real-time pollutant data from AQICN for a given city.

        Returns a dict with pollutant values (may have None for missing keys).
        Raises RuntimeError if the API call fails.
        """
        url = AQICN_API_URL.format(city=city, token=AQICN_API_TOKEN)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        if data.get("status") != "ok":
            msg = data.get("data", "Unknown error")
            raise RuntimeError(f"AQICN API error for '{city}': {msg}")

        iaqi = data["data"].get("iaqi", {})

        # Map AQICN keys → our feature names
        key_map = {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2":  "NO2",
            "so2":  "SO2",
            "co":   "CO",
            "o3":   "O3",
        }
        result = {}
        for api_key, feature in key_map.items():
            entry = iaqi.get(api_key)
            result[feature] = float(entry["v"]) if entry else None

        return result

    # ─────────────────────────────────────────────
    # Model metadata
    # ─────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._metadata.get("best_model_name", "Unknown")

    @property
    def r2_score(self) -> float:
        return self._metadata.get("r2_score", float("nan"))

    @property
    def rmse(self) -> float:
        return self._metadata.get("rmse", float("nan"))

    @property
    def mae(self) -> float:
        return self._metadata.get("mae", float("nan"))

    @property
    def metadata(self) -> dict:
        """Return a copy of the model metadata dict."""
        return dict(self._metadata)


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

def _cli():
    predictor = AQIPredictor()
    print("\n=== AQI Prediction System ===")
    print(f"Model : {predictor.model_name}")
    print(f"R²    : {predictor.r2_score:.4f}")
    print(f"RMSE  : {predictor.rmse:.2f}")
    print()
    print("Enter pollutant values (press Enter to skip / use imputed median):")

    data = {}
    for feat in FEATURE_NAMES:
        rng  = FEATURE_RANGES[feat]
        unit = rng["unit"]
        raw  = input(f"  {feat} [{unit}, 0-{rng['max']}]: ").strip()
        if raw:
            try:
                data[feat] = float(raw)
            except ValueError:
                print(f"    Invalid value for {feat}, will use imputed median.")

    try:
        result = predictor.predict(data)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        return

    print(f"\n{'=' * 45}")
    print(f"Predicted AQI  : {result['AQI']}")
    print(f"Category       : {result['Emoji']} {result['Category']}")
    print(f"Description    : {result['Description']}")
    print(f"Confidence     : {result['Confidence']}")
    print("\nHealth Advice:")
    for line in result["Health_Advice"]:
        print(f"  {line}")
    print("=" * 45)


if __name__ == "__main__":
    _cli()

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
    AQICN_API_TOKEN, AQICN_API_URL, MODELS_DIR,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# US EPA AQI sub-index → raw concentration breakpoints
# Format: (AQI_lo, AQI_hi, Conc_lo, Conc_hi)
# These are used to convert AQICN iaqi values (which are AQI sub-indices)
# back to raw pollutant concentrations expected by the ML model.
# ─────────────────────────────────────────────

_EPA_PM25_BP = [          # 24-hr avg, μg/m³
    (0,   50,  0.0,   12.0),
    (51,  100, 12.1,  35.4),
    (101, 150, 35.5,  55.4),
    (151, 200, 55.5,  150.4),
    (201, 300, 150.5, 250.4),
    (301, 400, 250.5, 350.4),
    (401, 500, 350.5, 500.4),
]

_EPA_PM10_BP = [          # 24-hr avg, μg/m³
    (0,   50,  0,   54),
    (51,  100, 55,  154),
    (101, 150, 155, 254),
    (151, 200, 255, 354),
    (201, 300, 355, 424),
    (301, 400, 425, 504),
    (401, 500, 505, 604),
]

_EPA_NO2_BP = [           # 1-hr avg, ppb → convert to μg/m³ (×1.88)
    (0,   50,  0,    53),
    (51,  100, 54,   100),
    (101, 150, 101,  360),
    (151, 200, 361,  649),
    (201, 300, 650,  1249),
    (301, 400, 1250, 1649),
    (401, 500, 1650, 2049),
]

_EPA_SO2_BP = [           # 1-hr avg, ppb → convert to μg/m³ (×2.62)
    (0,   50,  0,   35),
    (51,  100, 36,  75),
    (101, 150, 76,  185),
    (151, 200, 186, 304),
    (201, 300, 305, 604),
    (301, 400, 605, 804),
    (401, 500, 805, 1004),
]

_EPA_CO_BP = [            # 8-hr avg, ppm → convert to mg/m³ (×1.145)
    (0,   50,  0.0,  4.4),
    (51,  100, 4.5,  9.4),
    (101, 150, 9.5,  12.4),
    (151, 200, 12.5, 15.4),
    (201, 300, 15.5, 30.4),
    (301, 400, 30.5, 40.4),
    (401, 500, 40.5, 50.4),
]

_EPA_O3_BP = [            # 8-hr avg, ppb → convert to μg/m³ (×1.96)
    (0,   50,  0,   54),
    (51,  100, 55,  70),
    (101, 150, 71,  85),
    (151, 200, 86,  105),
    (201, 300, 106, 200),
    (301, 400, 201, 404),
    (401, 500, 405, 604),
]


def _interp_bp(val: float, breakpoints: list) -> float:
    """Linearly interpolate a raw concentration from a US EPA AQI sub-index."""
    for aqi_lo, aqi_hi, conc_lo, conc_hi in breakpoints:
        if aqi_lo <= val <= aqi_hi:
            if aqi_hi == aqi_lo:
                return float(conc_lo)
            return conc_lo + (val - aqi_lo) / (aqi_hi - aqi_lo) * (conc_hi - conc_lo)
    return float(breakpoints[-1][3])  # clamp to max


def _iaqi_to_concentrations(iaqi: dict) -> dict:
    """
    Convert AQICN individual AQI (iaqi) sub-index values into raw pollutant
    concentrations in the units the ML model was trained on:
      PM2.5, PM10, NO2, SO2 → μg/m³
      CO                    → mg/m³
      O3                    → μg/m³

    The AQICN API's ``iaqi`` field stores US-EPA AQI sub-index values, not
    raw concentrations.  Passing them directly to a model trained on raw
    concentration data (Indian CPCB city_day.csv) causes a large over-
    estimation of AQI (e.g. PM2.5 sub-index 152 being treated as 152 μg/m³
    instead of the correct ~57 μg/m³).
    """
    result = {}

    entry = iaqi.get("pm25")
    if entry is not None:
        result["PM2.5"] = round(_interp_bp(float(entry["v"]), _EPA_PM25_BP), 2)

    entry = iaqi.get("pm10")
    if entry is not None:
        result["PM10"] = round(_interp_bp(float(entry["v"]), _EPA_PM10_BP), 2)

    entry = iaqi.get("no2")
    if entry is not None:
        ppb = _interp_bp(float(entry["v"]), _EPA_NO2_BP)
        result["NO2"] = round(ppb * 1.88, 2)   # ppb → μg/m³

    entry = iaqi.get("so2")
    if entry is not None:
        ppb = _interp_bp(float(entry["v"]), _EPA_SO2_BP)
        result["SO2"] = round(ppb * 2.62, 2)   # ppb → μg/m³

    entry = iaqi.get("co")
    if entry is not None:
        ppm = _interp_bp(float(entry["v"]), _EPA_CO_BP)
        result["CO"] = round(ppm * 1.145, 3)   # ppm → mg/m³

    entry = iaqi.get("o3")
    if entry is not None:
        ppb = _interp_bp(float(entry["v"]), _EPA_O3_BP)
        result["O3"] = round(ppb * 1.96, 2)    # ppb → μg/m³

    return result


class AQIPredictor:
    """
    Encapsulates loading the trained model + preprocessing objects
    and exposes simple prediction / categorisation methods.
    """

    def __init__(self):
        self._model   = None
        self._scaler  = None
        self._imputer = None
        self._feature_normalizer = None
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

        # Load feature normalizer produced by step2_model_training_v2.py
        normalizer_path = os.path.join(MODELS_DIR, "feature_normalizer.pkl")
        if os.path.exists(normalizer_path):
            self._feature_normalizer = joblib.load(normalizer_path)

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

        # Apply feature normalizer (MinMaxScaler) if available (step2_model_training_v2.py)
        if self._feature_normalizer is not None:
            row_scaled = self._feature_normalizer.transform(row_scaled)

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

        # AQICN iaqi values are US-EPA AQI sub-indices, not raw concentrations.
        # Convert them to the raw concentration units the ML model was trained on.
        concentrations = _iaqi_to_concentrations(iaqi)

        # Build result with None for any pollutant missing from the API response
        return {feat: concentrations.get(feat) for feat in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]}

    # ─────────────────────────────────────────────
    # Model metadata
    # ─────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._metadata.get("best_model_name", "Unknown")

    @property
    def r2_score(self) -> float:
        return self._metadata.get("r2_score", self._metadata.get("test_r2", float("nan")))

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

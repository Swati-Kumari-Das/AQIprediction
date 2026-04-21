"""
Configuration file for AQI Prediction System
"""

# ─────────────────────────────────────────────
# AQICN API Configuration
# ─────────────────────────────────────────────
# Replace with your own token from https://aqicn.org/api/
# Using "demo" is only suitable for quick testing and has very low rate limits.
AQICN_API_TOKEN = "demo"
AQICN_API_URL = "https://api.waqi.info/feed/{city}/?token={token}"

# ─────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────
FEATURE_NAMES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET_COLUMN = "AQI"

# Valid input ranges for each pollutant (used for validation)
FEATURE_RANGES = {
    "PM2.5": {"min": 0.0,  "max": 1000.0, "unit": "µg/m³"},
    "PM10":  {"min": 0.0,  "max": 1500.0, "unit": "µg/m³"},
    "NO2":   {"min": 0.0,  "max": 500.0,  "unit": "µg/m³"},
    "SO2":   {"min": 0.0,  "max": 500.0,  "unit": "µg/m³"},
    "CO":    {"min": 0.0,  "max": 100.0,  "unit": "mg/m³"},
    "O3":    {"min": 0.0,  "max": 500.0,  "unit": "µg/m³"},
}

# ─────────────────────────────────────────────
# AQI Category Thresholds (India standard)
# ─────────────────────────────────────────────
AQI_CATEGORIES = [
    {
        "name":        "Good",
        "range":       (0, 50),
        "color":       "#00e400",
        "text_color":  "#000000",
        "emoji":       "😊",
        "description": "Air quality is satisfactory and poses little or no risk.",
    },
    {
        "name":        "Satisfactory",
        "range":       (51, 100),
        "color":       "#92d050",
        "text_color":  "#000000",
        "emoji":       "🙂",
        "description": "Air quality is acceptable. Minor discomfort for sensitive people.",
    },
    {
        "name":        "Moderate",
        "range":       (101, 200),
        "color":       "#ffff00",
        "text_color":  "#000000",
        "emoji":       "😐",
        "description": "Sensitive individuals may experience health effects.",
    },
    {
        "name":        "Poor",
        "range":       (201, 300),
        "color":       "#ff7e00",
        "text_color":  "#ffffff",
        "emoji":       "😷",
        "description": "Health effects for everyone; serious effects for sensitive groups.",
    },
    {
        "name":        "Very Poor",
        "range":       (301, 400),
        "color":       "#ff0000",
        "text_color":  "#ffffff",
        "emoji":       "🤢",
        "description": "Health alert: Everyone may experience serious health effects.",
    },
    {
        "name":        "Severe",
        "range":       (401, float("inf")),
        "color":       "#7e0023",
        "text_color":  "#ffffff",
        "emoji":       "☠️",
        "description": "Emergency conditions. Entire population is affected.",
    },
]

# ─────────────────────────────────────────────
# Health Advice per AQI Category
# ─────────────────────────────────────────────
HEALTH_ADVICE = {
    "Good": [
        "✅ Air quality is great — enjoy outdoor activities!",
        "✅ No precautions needed for the general public.",
        "✅ Great day for exercise outdoors.",
    ],
    "Satisfactory": [
        "🟢 Air quality is generally acceptable.",
        "⚠️ Very sensitive individuals should consider reducing prolonged outdoor exertion.",
        "✅ Fine for most people to be outdoors.",
    ],
    "Moderate": [
        "⚠️ Children and elderly should limit prolonged outdoor activities.",
        "⚠️ People with respiratory or heart conditions should be cautious.",
        "😷 Consider wearing a mask if spending extended time outdoors.",
        "🏠 Keep windows closed during peak traffic hours.",
    ],
    "Poor": [
        "🚫 Everyone should reduce outdoor physical activity.",
        "😷 Wear N95/N99 masks when going outside.",
        "🏠 Keep windows and doors closed.",
        "⚠️ Avoid morning/evening outdoor exercise.",
        "💊 Sensitive groups should consult a doctor if experiencing symptoms.",
    ],
    "Very Poor": [
        "🚫 Avoid all unnecessary outdoor activities.",
        "😷 Use high-quality air purifiers indoors.",
        "🏠 Stay indoors as much as possible.",
        "🚑 Seek medical attention if you experience breathing difficulty.",
        "⚠️ Children, elderly, and pregnant women should not go outside.",
    ],
    "Severe": [
        "🆘 Emergency conditions — do NOT go outside.",
        "🏠 Seal windows and doors to prevent outdoor air entering.",
        "😷 Use N99/P100 masks if you must go out.",
        "🚑 Seek immediate medical care for any respiratory distress.",
        "🚫 All outdoor sports and activities cancelled.",
        "⚠️ Schools and offices should consider closure.",
    ],
}

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────
DATASET_PATH = "city_day.csv"
OUTPUT_DIR = "output"
MODELS_DIR = "models"

SCALER_PATH        = f"{OUTPUT_DIR}/scaler.pkl"
IMPUTER_PATH       = f"{OUTPUT_DIR}/imputer.pkl"
X_TRAIN_PATH       = f"{OUTPUT_DIR}/X_train_processed.pkl"
X_TEST_PATH        = f"{OUTPUT_DIR}/X_test_processed.pkl"
Y_TRAIN_PATH       = f"{OUTPUT_DIR}/y_train.pkl"
Y_TEST_PATH        = f"{OUTPUT_DIR}/y_test.pkl"
DATA_STATS_PATH    = f"{OUTPUT_DIR}/data_stats.json"
PREPROC_RPT_PATH   = f"{OUTPUT_DIR}/preprocessing_report.txt"

BEST_MODEL_PATH    = f"{MODELS_DIR}/best_model.pkl"
MODEL_META_PATH    = f"{MODELS_DIR}/model_metadata.json"
FEAT_IMP_PATH      = f"{MODELS_DIR}/feature_importance.csv"
TRAINING_RPT_PATH  = f"{MODELS_DIR}/training_report.txt"

# ─────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5

# Model hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}

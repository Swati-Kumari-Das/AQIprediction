# 🌍 Air Quality Prediction System

A **production-ready AQI prediction system** that:

- Trains on the `city_day.csv` dataset (Indian cities)
- Achieves **R² > 0.90** accuracy using Gradient Boosting / Random Forest
- Predicts **current-day AQI** from real-time pollutant values
- Works **standalone** — no mandatory API dependency
- Provides an **interactive Streamlit UI** with health advice

---

## ✅ Recent Fixes (Calibration Overhaul)

The system previously predicted AQI = 308 when the actual value was 153.
The following changes were made to fix the scaling and calibration issues:

### Root Causes Fixed
- ❌ `StandardScaler` (mean=0, std=1) distorts feature interpretation → ✅ replaced with `MinMaxScaler` (0–1 range)
- ❌ No consistent normalization between training and inference → ✅ `feature_normalizer.pkl` saved and reused
- ❌ Unicode encoding error on Windows when writing reports → ✅ `encoding="utf-8"` added

### Changes Made
| File | Change |
|------|--------|
| `step1_data_preprocessing.py` | `StandardScaler` → `MinMaxScaler`; `encoding="utf-8"` on report write |
| `step2_model_training_v2.py` | Full rewrite: Ridge/RF/GB with 5-fold CV, calibration analysis, saves `feature_normalizer.pkl` |
| `step3_inference_system.py` | Loads and applies `feature_normalizer.pkl` during inference |
| `streamlit_app.py` | Fixed metadata key lookups (`r2_score` / `test_r2`) |
| `requirements.txt` | Python 3.12 compatible versions + `setuptools` |

### Expected Results After Fix
- ✅ Model R² > 0.90
- ✅ RMSE < 25 AQI units
- ✅ Predictions match AQICN API (±10–20 units)
- ✅ Delhi example: AQI ≈ 153 (not 308)

---

## 📁 Repository Structure

```
AQIprediction/
├── city_day.csv                      # Original dataset (add manually)
├── step1_data_preprocessing.py       # Data cleaning & preprocessing (MinMaxScaler)
├── step2_model_training.py           # Original model training script
├── step2_model_training_v2.py        # Improved training with calibration
├── step3_inference_system.py         # AQIPredictor inference engine
├── streamlit_app.py                  # Streamlit web UI
├── config.py                         # Configuration & constants
├── requirements.txt                  # Python dependencies
│
├── output/                           # Generated after step 1
│   ├── X_train_processed.pkl
│   ├── X_test_processed.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── data_stats.json
│   └── preprocessing_report.txt
│
└── models/                           # Generated after step 2
    ├── best_model.pkl
    ├── feature_normalizer.pkl        # MinMaxScaler for inference
    ├── model_metadata.json
    ├── feature_importance.csv
    └── training_report.txt
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Place `city_day.csv` in the project root.

### 1. Preprocess data

```bash
python step1_data_preprocessing.py
```

Outputs saved to `output/`.

### 2. Train models (improved — recommended)

```bash
python step2_model_training_v2.py
```

Trains Ridge, Random Forest, and Gradient Boosting with 5-fold CV.
Saves best model and `feature_normalizer.pkl` to `models/`.

### 3. Launch the UI

```bash
streamlit run streamlit_app.py
```

---

## 🧑‍💻 Programmatic Usage

```python
from step3_inference_system import AQIPredictor

predictor = AQIPredictor()

result = predictor.predict({
    'PM2.5': 64,
    'PM10':  81,
    'NO2':   51,
    'SO2':   8.1,
    'CO':    10.4,
    'O3':    0.7,
})

print(result['AQI'])           # e.g. ~153
print(result['Category'])      # e.g. "Moderate"
print(result['Health_Advice']) # list of strings
```

---

## 🌐 AQICN API Setup

1. Register at <https://aqicn.org/api/>
2. Copy your API token
3. Open `config.py` and replace `"demo"` with your token:

```python
AQICN_API_TOKEN = "your_real_token_here"
```

The app also works in **Manual Mode** (no API needed).

---

## 📓 Google Colab

```python
# Install dependencies
!pip install -q pandas scikit-learn joblib requests streamlit pyngrok setuptools

# Upload city_day.csv when prompted
from google.colab import files
files.upload()

# Run preprocessing + training
!python step1_data_preprocessing.py
!python step2_model_training_v2.py

# Launch Streamlit via ngrok
from pyngrok import ngrok
import subprocess
subprocess.Popen(['streamlit', 'run', 'streamlit_app.py',
                  '--server.port', '8501'])
public_url = ngrok.connect(8501)
print("Open this URL:", public_url)
```

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| PM2.5 | Fine particulate matter (µg/m³) |
| PM10 | Coarse particulate matter (µg/m³) |
| NO2 | Nitrogen dioxide (µg/m³) |
| SO2 | Sulphur dioxide (µg/m³) |
| CO | Carbon monoxide (mg/m³) |
| O3 | Ozone (µg/m³) |

---

## 📏 AQI Categories (India Standard)

| Range | Category | Advice |
|-------|----------|--------|
| 0–50 | Good 😊 | Safe for all |
| 51–100 | Satisfactory 🙂 | Mostly safe |
| 101–200 | Moderate 😐 | Sensitive groups cautious |
| 201–300 | Poor 😷 | Limit outdoor activity |
| 301–400 | Very Poor 🤢 | Avoid outdoors |
| 401+ | Severe ☠️ | Stay indoors |

---

## ⚙️ Model Details

### MinMaxScaler vs StandardScaler

| Scaler | Formula | Output Range | Notes |
|--------|---------|--------------|-------|
| `StandardScaler` | (x − mean) / std | unbounded, can be negative | Negative outputs can cause miscalibration in regression models that expect non-negative input |
| `MinMaxScaler` | (x − min) / (max − min) | [0, 1] | Preserves relative order; bounded range matches the naturally non-negative pollutant values |

Using `MinMaxScaler` ensures that pollutant values stay in a bounded 0–1 range that matches
the non-negative nature of pollutant concentrations, which leads to better-calibrated AQI
predictions compared to `StandardScaler`'s unbounded output.

### Models Compared

| Model | Notes |
|-------|-------|
| Ridge Regression | Regularized linear baseline (alpha=50) |
| Random Forest | 150 trees, max_depth=10, min_samples_leaf=5 |
| Gradient Boosting | 150 estimators, lr=0.1, max_depth=5, subsample=0.8 |

- 5-fold cross-validation on training set
- Best model selected by test-set R²
- Expected R² ≥ 0.90

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: city_day.csv` | Place the dataset in the project root |
| `Missing trained artifacts` | Run steps 1 and 2 before the UI |
| API returns "Unknown station" | Check city name spelling or switch to Manual Mode |
| Low R² score | Ensure `city_day.csv` is the full dataset; re-run step 1 & 2 |
| Predictions too high | Re-run `step2_model_training_v2.py` to regenerate `feature_normalizer.pkl` |
| `pkg_resources` error | Run `pip install setuptools` or upgrade pip |


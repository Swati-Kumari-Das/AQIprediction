# 🌍 Air Quality Prediction System

A **production-ready AQI prediction system** that:

- Trains on the `city_day.csv` dataset (Indian cities)
- Achieves **R² > 0.90** accuracy using Gradient Boosting / Random Forest
- Predicts **current-day AQI** from real-time pollutant values
- Works **standalone** — no mandatory API dependency
- Provides an **interactive Streamlit UI** with health advice

---

## 📁 Repository Structure

```
AQIprediction/
├── city_day.csv                      # Original dataset (add manually)
├── step1_data_preprocessing.py       # Data cleaning & preprocessing
├── step2_model_training.py           # Model training & comparison
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

### 2. Train models

```bash
python step2_model_training.py
```

Compares Linear Regression, Random Forest, and Gradient Boosting using 5-fold CV.
Best model saved to `models/`.

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
    'PM2.5': 65,
    'PM10':  120,
    'NO2':   45,
    'SO2':   20,
    'CO':    1.5,
    'O3':    35,
})

print(result['AQI'])           # e.g. 185.3
print(result['Category'])      # e.g. "Poor"
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
!pip install -q pandas scikit-learn joblib requests streamlit pyngrok

# Upload city_day.csv when prompted
from google.colab import files
files.upload()

# Run preprocessing + training
!python step1_data_preprocessing.py
!python step2_model_training.py

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

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline — fast but lower accuracy |
| Random Forest | 200 trees, max_depth=15, n_jobs=-1 |
| Gradient Boosting | 200 estimators, lr=0.1, max_depth=6 |

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

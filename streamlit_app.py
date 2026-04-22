"""
Streamlit UI for the AQI Prediction System
==========================================
Run:
    streamlit run streamlit_app.py

For Google Colab:
    !pip install -q streamlit pyngrok
    from pyngrok import ngrok
    !streamlit run streamlit_app.py &
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    FEATURE_NAMES, FEATURE_RANGES,
    AQI_CATEGORIES, HEALTH_ADVICE,
    MODEL_META_PATH, FEAT_IMP_PATH,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0 0.25rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .aqi-box {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .aqi-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .aqi-category {
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .aqi-desc {
        font-size: 0.95rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    .advice-item {
        padding: 0.35rem 0;
        font-size: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Load predictor (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_predictor():
    from step3_inference_system import AQIPredictor  # noqa: PLC0415
    return AQIPredictor()


def try_load_predictor():
    """Return (predictor, error_message)."""
    try:
        return load_predictor(), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error loading model: {exc}"


# ─────────────────────────────────────────────
# AQI display helper
# ─────────────────────────────────────────────

def render_aqi_result(result: dict):
    """Render the coloured AQI result card + advice."""
    color      = result["Color"]
    text_color = "#ffffff" if result["AQI"] > 100 else "#000000"

    st.markdown(
        f"""
        <div class="aqi-box" style="background:{color}; color:{text_color};">
            <div class="aqi-value">{result['AQI']}</div>
            <div class="aqi-category">{result['Emoji']} {result['Category']}</div>
            <div class="aqi-desc">{result['Description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conf_color = {"High": "green", "Medium": "orange", "Low": "red"}.get(
        result["Confidence"], "grey"
    )
    st.markdown(
        f"**Prediction Confidence:** "
        f"<span style='color:{conf_color};font-weight:700;'>"
        f"{result['Confidence']}</span>",
        unsafe_allow_html=True,
    )

    st.subheader("🩺 Health Advice")
    for line in result["Health_Advice"]:
        st.markdown(f'<div class="advice-item">{line}</div>', unsafe_allow_html=True)

    # Pollutant bar chart
    input_vals = {
        k: v for k, v in result["Input_Values"].items()
        if not np.isnan(v)
    }
    if input_vals:
        st.subheader("📊 Pollutant Levels")
        df_vals = pd.DataFrame(
            {"Pollutant": list(input_vals.keys()),
             "Value":     list(input_vals.values())}
        ).set_index("Pollutant")
        st.bar_chart(df_vals)


# ─────────────────────────────────────────────
# Sidebar – AQI Legend
# ─────────────────────────────────────────────

def render_sidebar():
    st.sidebar.title("🎨 AQI Scale")
    for cat in AQI_CATEGORIES:
        lo, hi = cat["range"]
        hi_str = str(int(hi)) if hi != float("inf") else "500+"
        st.sidebar.markdown(
            f"<div style='background:{cat['color']};color:"
            f"{'#000' if lo < 200 else '#fff'};"
            f"border-radius:6px;padding:4px 8px;margin:2px 0;'>"
            f"{cat['emoji']} <b>{cat['name']}</b> ({lo}–{hi_str})</div>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tip:** Use manual mode if you already know the pollutant "
        "concentrations. Use API mode to fetch live data from AQICN."
    )


# ─────────────────────────────────────────────
# Tab 1 – Prediction
# ─────────────────────────────────────────────

def tab_prediction(predictor):
    st.markdown(
        '<div class="main-header">🌍 AQI Prediction System</div>'
        '<div class="sub-header">Predict Air Quality Index from real-time pollutant values</div>',
        unsafe_allow_html=True,
    )

    city = st.text_input("🏙️ City Name", placeholder="e.g. Delhi, Mumbai, Bangalore")

    mode = st.radio(
        "Input Mode",
        ["✏️ Manual Input", "🌐 Live API (AQICN)"],
        horizontal=True,
    )

    pollutants: dict[str, float | None] = {}

    if mode == "✏️ Manual Input":
        st.subheader("Enter Pollutant Concentrations")
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3, col1, col2, col3]
        for feat, col in zip(FEATURE_NAMES, cols):
            rng = FEATURE_RANGES[feat]
            with col:
                val = st.number_input(
                    f"{feat} ({rng['unit']})",
                    min_value=float(rng["min"]),
                    max_value=float(rng["max"]),
                    value=0.0,
                    step=0.1,
                    key=f"input_{feat}",
                )
                pollutants[feat] = val

        if st.button("🔍 Predict AQI", type="primary", use_container_width=True):
            try:
                result = predictor.predict(pollutants)
                if city:
                    st.markdown(f"### Results for **{city}**")
                render_aqi_result(result)
            except ValueError as exc:
                st.error(f"⚠️ Input Error: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Prediction Error: {exc}")

    else:  # API mode
        if not city:
            st.warning("⚠️ Please enter a city name to fetch live data.")
            return

        if st.button("🌐 Fetch & Predict AQI", type="primary", use_container_width=True):
            with st.spinner(f"Fetching live data for **{city}**…"):
                try:
                    pollutants = predictor.fetch_api_data(city)
                except RuntimeError as exc:
                    st.error(f"❌ API Error: {exc}")
                    st.info(
                        "💡 Switch to **Manual Input** mode to enter "
                        "pollutant values directly."
                    )
                    return

            # Show fetched values
            st.success("✅ Live data fetched successfully!")
            cols = st.columns(len(FEATURE_NAMES))
            for feat, col in zip(FEATURE_NAMES, cols):
                with col:
                    val = pollutants.get(feat)
                    display = f"{val:.1f}" if val is not None else "N/A"
                    rng = FEATURE_RANGES[feat]
                    st.metric(label=f"{feat} ({rng['unit']})", value=display)

            try:
                result = predictor.predict(pollutants)
                st.markdown(f"### Results for **{city}**")
                render_aqi_result(result)
            except ValueError as exc:
                st.error(f"⚠️ Input Error: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Prediction Error: {exc}")


# ─────────────────────────────────────────────
# Tab 2 – Model Info
# ─────────────────────────────────────────────

def tab_model_info(predictor):
    st.header("📊 Model Information")

    meta = predictor.metadata

    if not meta:
        st.warning("Model metadata not found. Re-run training to generate it.")
        return

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Best Model",  meta.get("best_model_name", "N/A"))
    with c2:
        r2_val = meta.get("r2_score", meta.get("test_r2", 0))
        st.metric("R² Score",    f"{r2_val:.4f}")
    with c3:
        st.metric("RMSE",        f"{meta.get('rmse', 0):.2f}")
    with c4:
        st.metric("MAE",         f"{meta.get('mae', 0):.2f}")

    # Cross-validation
    st.subheader("Cross-Validation Results")
    st.info(
        f"5-Fold CV R²: **{meta.get('cv_r2_mean', 0):.4f}** "
        f"± {meta.get('cv_r2_std', 0):.4f}"
    )

    # All models comparison
    all_models = meta.get("all_models", {})
    if all_models:
        st.subheader("Model Comparison")
        rows = []
        for name, res in all_models.items():
            rows.append({
                "Model":       name,
                "R² (test)":   round(res.get("test_r2",   0), 4),
                "RMSE":        round(res.get("test_rmse", 0), 2),
                "MAE":         round(res.get("test_mae",  0), 2),
                "MAPE (%)":    round(res.get("test_mape", 0), 2),
                "CV R² (mean)": round(res.get("cv_r2_mean", 0), 4),
            })
        df_comp = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df_comp, use_container_width=True)

    # Feature importance
    if os.path.exists(FEAT_IMP_PATH):
        st.subheader("Feature Importance")
        fi_df = pd.read_csv(FEAT_IMP_PATH).set_index("Feature")
        st.bar_chart(fi_df["Importance"])
        st.dataframe(fi_df.reset_index(), use_container_width=True)


# ─────────────────────────────────────────────
# Tab 3 – About
# ─────────────────────────────────────────────

def tab_about():
    st.header("ℹ️ About this System")
    st.markdown(
        """
### 🌿 Air Quality Prediction System

This system predicts the **current Air Quality Index (AQI)** from real-time or
manually entered pollutant concentrations using a machine-learning model trained
on historical Indian city data.

---

#### 📂 Dataset
- **Source:** `city_day.csv` — daily AQI data for multiple Indian cities
- **Features used:** PM2.5, PM10, NO2, SO2, CO, O3
- **Target:** AQI (current day)

---

#### 🏗️ Architecture

| Step | File | Purpose |
|------|------|---------|
| 1 | `step1_data_preprocessing.py` | Clean data, impute, scale |
| 2 | `step2_model_training.py` | Train & compare 3 models |
| 3 | `step3_inference_system.py` | `AQIPredictor` class |
| 4 | `streamlit_app.py` | This UI |

---

#### 📏 AQI Categories (India Standard)

| Range | Category | Description |
|-------|----------|-------------|
| 0–50 | Good | Safe for everyone |
| 51–100 | Satisfactory | Minor discomfort for very sensitive people |
| 101–200 | Moderate | Sensitive individuals should be cautious |
| 201–300 | Poor | Everyone may experience health effects |
| 301–400 | Very Poor | Serious health effects for all |
| 401+ | Severe | Emergency conditions |

---

#### 🚀 Running Locally

```bash
# Step 1: Download dataset city_day.csv
# Step 2: Preprocess data
python step1_data_preprocessing.py

# Step 3: Train models
python step2_model_training.py

# Step 4: Launch UI
streamlit run streamlit_app.py
```

#### 🧪 Running in Google Colab

```python
!pip install -q streamlit pyngrok pandas scikit-learn joblib requests
!python step1_data_preprocessing.py
!python step2_model_training.py

from pyngrok import ngrok
import subprocess
proc = subprocess.Popen(['streamlit', 'run', 'streamlit_app.py'])
public_url = ngrok.connect(8501)
print(public_url)
```
        """
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    render_sidebar()

    predictor, err = try_load_predictor()

    if err:
        st.error(f"❌ Could not load the trained model.\n\n{err}")
        st.info(
            "**Please run the training pipeline first:**\n\n"
            "```bash\n"
            "python step1_data_preprocessing.py\n"
            "python step2_model_training.py\n"
            "```"
        )
        tab_about()
        return

    tab1, tab2, tab3 = st.tabs(
        ["🔮 AQI Prediction", "📊 Model Info", "ℹ️ About"]
    )

    with tab1:
        tab_prediction(predictor)

    with tab2:
        tab_model_info(predictor)

    with tab3:
        tab_about()


if __name__ == "__main__":
    main()

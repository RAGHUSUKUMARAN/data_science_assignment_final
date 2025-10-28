# -------- streamlit_app.py (drop-in) --------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Titanic Survival Predictor")

# ==== CONFIG ====
# Change these paths/names to your actual saved files
BASE_DIR = Path(r"D:\DATA-SCIENCE\ASSIGNMENTS\7 logistic regression")
MODEL_PATH  = BASE_DIR / "model.pkl"           # e.g., your trained LogisticRegression
SCALER_PATH = BASE_DIR / "scaler.pkl"          # e.g., your StandardScaler

# These are the numeric columns you scaled during training
NUM_COLS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# If you one-hot-encoded categorical features during training,
# list all final feature names the model was fit on (optional but best).
MODEL_FEATURES = None   # set to a list if you know it, else we infer with model.feature_names_in_

# ==== HELPERS ====
@st.cache_resource
def load_artifacts(model_path: Path, scaler_path: Path):
    model, scaler = None, None
    if model_path.exists():
        model = joblib.load(model_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    return model, scaler

def safe_get_feature_names(estimator, fallback=None):
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return fallback

# ==== LOAD ====
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

with st.expander("ℹ️ Artifacts status", expanded=False):
    st.write("- Model loaded:", model is not None, f"({MODEL_PATH.name})")
    st.write("- Scaler loaded:", scaler is not None, f"({SCALER_PATH.name})")

if model is None:
    st.error(f"Could not load model from: {MODEL_PATH}")
    st.stop()

# ==== INPUT UI ====
st.subheader("Enter Passenger Details")

col1, col2 = st.columns(2)
with col1:
    Pclass = st.selectbox("Ticket Class (1=Upper, 2=Middle, 3=Lower)", [1, 2, 3], index=1)
    Age    = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
    SibSp  = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
with col2:
    Parch  = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
    Fare   = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.5)

# If you used additional engineered/one-hot columns during training, add those inputs here
# and then build them into input_df below so columns line up.

# Build the raw input dataframe
input_df = pd.DataFrame([{
    "Pclass": Pclass,
    "Age": Age,
    "SibSp": SibSp,
    "Parch": Parch,
    "Fare": Fare,
}])

st.write("**Raw input**")
st.dataframe(input_df)

# ---- Scaling + Prediction block ----
predict_btn = st.button("Predict Survival", type="primary")

if predict_btn:
    try:
        # 1) Ensure numeric columns exist
        for c in NUM_COLS:
            if c not in input_df.columns:
                raise KeyError(f"Missing numeric column in input: {c}")

        # 2) Scale numeric columns if scaler is available
        if scaler is not None:
            try:
                input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])
            except ValueError:
                # align by names if scaler expects different ordering
                st.warning("Scaler rejected input shape/order — attempting to auto-align features...")
                scaler_cols = safe_get_feature_names(scaler, fallback=NUM_COLS)
                # Add missing columns
                for c in scaler_cols:
                    if c not in input_df.columns:
                        input_df[c] = 0
                input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
                st.info(f"Used scaler columns: {scaler_cols}")
        else:
            st.warning("No scaler found — proceeding without scaling.")

        # 3) Align to model's expected columns/order
        model_cols = MODEL_FEATURES or safe_get_feature_names(model, fallback=list(input_df.columns))
        # Ensure all required columns exist
        for c in model_cols:
            if c not in input_df.columns:
                input_df[c] = 0
        input_for_model = input_df[model_cols]

        # 4) Predict
        pred = int(model.predict(input_for_model)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_for_model)[0][1])

        # 5) Display nicely
        label = "✅ Survived" if pred == 1 else "❌ Did not survive"
        if proba is not None:
            st.success(f"Prediction: **{label}** — survival probability: **{proba:.2%}**")
        else:
            st.success(f"Prediction: **{label}**")
        st.write("### Input used for prediction (after scaling / alignment)")
        st.dataframe(input_for_model)

    except Exception as e:
        st.error("An error occurred during scaling/prediction.")
        st.exception(e)
        # helpful debug info:
        scaler_names = safe_get_feature_names(scaler)
        model_names = safe_get_feature_names(model)
        st.write("**Debug info:**")
        st.write("- Input columns:", list(input_df.columns))
        st.write("- Scaler.feature_names_in_:", scaler_names if scaler_names else "N/A")
        st.write("- Model.feature_names_in_:", model_names if model_names else "N/A")
# -------- end file --------

"""
Streamlit frontend for the diabetes risk prediction pipeline.

Run with:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

# Make src/ importable regardless of the working directory streamlit is launched from
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import GENDER_OPTIONS, SMOKING_OPTIONS  # noqa: E402
from predict import load_pipeline, predict_one  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def get_pipeline():
    return load_pipeline()


def get_model_name() -> str:
    info_path = MODELS_DIR / "model_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text()).get("best_model", "Unknown")
    return "Unknown"


st.title("🩺 Diabetes Risk Predictor")
st.caption(
    f"Predicts diabetes risk from basic health indicators using a trained "
    f"**{get_model_name()}** model."
)

try:
    pipe = get_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.form("prediction_form"):
    st.subheader("Patient information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        hba1c = st.number_input(
            "HbA1c level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1
        )
        glucose = st.number_input(
            "Blood glucose level (mg/dL)", min_value=50, max_value=400, value=100, step=1
        )
        smoking_history = st.selectbox("Smoking history", SMOKING_OPTIONS)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "gender": gender,
        "smoking_history": smoking_history,
    }

    result = predict_one(pipe, input_dict)

    st.divider()
    st.subheader("Result")

    if result["prediction"] == 1:
        st.error(f"⚠️ **{result['label']}**")
    else:
        st.success(f"✅ **{result['label']}**")

    if result["probability"] is not None:
        st.metric("Estimated probability of diabetes", f"{result['probability']:.1%}")
        st.progress(min(max(result["probability"], 0.0), 1.0))

    st.caption(
        "This is a statistical estimate from a machine learning model trained on a "
        "public dataset, not a medical diagnosis. Consult a healthcare professional "
        "for medical advice."
    )

with st.expander("Model performance (test set)"):
    metrics_path = METRICS_DIR / "results.json"
    if metrics_path.exists():
        all_results = json.loads(metrics_path.read_text())
        best_model = get_model_name()
        if best_model in all_results:
            m = all_results[best_model]["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.1%}")
            c2.metric("Precision", f"{m['precision']:.1%}")
            c3.metric("Recall", f"{m['recall']:.1%}")
            c4.metric("F1 score", f"{m['f1']:.1%}")
            if "roc_auc" in m:
                st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    else:
        st.write("No metrics found. Run `python src/train.py` to generate them.")

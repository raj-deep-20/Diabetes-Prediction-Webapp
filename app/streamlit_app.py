import json
import sys
from pathlib import Path

import streamlit as st

# Make src/ importable regardless of the working directory Streamlit is launched from.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import GENDER_OPTIONS, SMOKING_OPTIONS  # noqa: E402
from predict import load_pipeline, predict_one  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained clinical palette with stronger hierarchy and comfortable spacing.
st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --surface: #ffffff;
        --soft: #f5f8ff;
        --line: #e6eaf2;
        --success: #087443;
        --danger: #b42318;
    }
    .stApp { background: #f7f9fc; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,249,252,.88); }
    .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #101828; }
    [data-testid="stSidebar"] * { color: #eef4ff; }
    [data-testid="stSidebar"] .stCaption { color: #b8c4d9; }
    h1, h2, h3 { letter-spacing: -0.025em; color: var(--ink); }
    h1 { font-size: clamp(2rem, 4vw, 3.25rem) !important; margin-bottom: .4rem; }
    h2 { font-size: 1.45rem !important; }
    .hero { padding: 1.8rem 2rem; border: 1px solid #dbe6ff; border-radius: 22px;
            background: linear-gradient(135deg, #eef4ff 0%, #ffffff 62%);
            box-shadow: 0 10px 28px rgba(31, 62, 114, .07); margin-bottom: 1.4rem; }
    .eyebrow { color: var(--primary); font-weight: 800; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; }
    .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 720px; line-height: 1.65; }
    .section-card { background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
                    padding: 1.25rem 1.35rem .55rem; box-shadow: 0 5px 18px rgba(16,24,40,.04); }
    .section-label { color: var(--ink); font-size: 1.08rem; font-weight: 750; margin-bottom: .15rem; }
    .section-help { color: var(--muted); font-size: .88rem; margin-bottom: .9rem; }
    .result-card { border-radius: 18px; padding: 1.3rem 1.45rem; margin-top: .8rem; border: 1px solid; }
    .result-card.high { background: #fff6f5; border-color: #fecdca; }
    .result-card.low { background: #f0fdf7; border-color: #a6f4c5; }
    .result-title { font-size: 1.35rem; font-weight: 800; margin-bottom: .2rem; }
    .result-note { color: var(--muted); margin: 0; }
    .disclaimer { font-size: .82rem; color: var(--muted); background: #f8fafc; border-left: 3px solid #98a2b3;
                  border-radius: 4px; padding: .8rem 1rem; line-height: 1.55; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: .85rem 1rem; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button { border-radius: 10px; font-weight: 750; }
    div[data-testid="stFormSubmitButton"] button { min-height: 3rem; background: var(--primary); border: 0; }
    div[data-testid="stFormSubmitButton"] button:hover { background: var(--primary-dark); }
    .stProgress > div > div { background: var(--primary); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return load_pipeline()


def get_model_name() -> str:
    info_path = MODELS_DIR / "model_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text()).get("best_model", "Unknown")
    return "Unknown"


def yes_no(value: str) -> int:
    return int(value == "Yes")


# Sidebar keeps supporting information available without competing with the main task.
with st.sidebar:
    st.markdown("## 🩺 Health insights")
    st.caption("A simple, privacy-conscious interface for exploring a model-based estimate.")
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "Enter the available health indicators, submit the form, and review the model’s "
        "estimated probability. Use the result as a conversation starter—not a diagnosis."
    )
    st.markdown("### Input tips")
    st.markdown(
        "Use current measurements where possible. If a value is unavailable, avoid guessing; "
        "the model is only as meaningful as the information provided."
    )
    st.divider()
    st.caption(f"Active model: **{get_model_name()}**")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine-learning health screening</div>
      <h1>Diabetes Risk Predictor</h1>
      <p class="hero-copy">Explore an estimated diabetes risk from a small set of routine health indicators, presented in a clear and approachable format.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    pipe = get_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patient information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-help">Provide the indicators used by the trained prediction pipeline.</div>', unsafe_allow_html=True)

    with st.form("prediction_form", border=False):
        demographics, clinical = st.columns(2, gap="large")
        with demographics:
            st.markdown("**Demographics**")
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0, help="Age in years.")
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            smoking_history = st.selectbox("Smoking history", SMOKING_OPTIONS)
        with clinical:
            st.markdown("**Clinical indicators**")
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1, help="Body mass index.")
            hba1c = st.number_input("HbA1c level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1, help="Glycated hemoglobin level.")
            glucose = st.number_input("Blood glucose (mg/dL)", min_value=50, max_value=400, value=100, step=1, help="Blood glucose measurement.")
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart disease", ["No", "Yes"])

        submitted = st.form_submit_button("Generate risk estimate", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("### What you’ll receive")
    st.markdown(
        "The model returns a classification and, when available, a probability score. "
        "The score is an estimate based on the model and should be interpreted alongside professional clinical guidance."
    )
    st.info("Your entries are used for this prediction session and are not displayed as a patient record.")

if submitted:
    input_dict = {
        "age": age,
        "hypertension": yes_no(hypertension),
        "heart_disease": yes_no(heart_disease),
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "gender": gender,
        "smoking_history": smoking_history,
    }
    result = predict_one(pipe, input_dict)
    probability = result.get("probability")
    is_high = result["prediction"] == 1

    st.divider()
    st.markdown("### Your estimate")
    st.markdown(
        f'<div class="result-card {"high" if is_high else "low"}">'
        f'<div class="result-title">{"⚠️ " if is_high else "✅ "}{result["label"]}</div>'
        f'<p class="result-note">{"The model flags a higher estimated risk based on the provided indicators." if is_high else "The model does not flag a higher estimated risk based on the provided indicators."}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    if probability is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Estimated probability", f"{probability:.1%}")
        with c2:
            st.markdown("**Probability scale**")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption("This visual reflects the model output; it is not a clinical risk threshold.")

    st.markdown(
        '<div class="disclaimer"><strong>Important:</strong> This is a statistical estimate from a machine-learning model trained on a public dataset, not a medical diagnosis. Consult a qualified healthcare professional for medical advice or follow-up.</div>',
        unsafe_allow_html=True,
    )

with st.expander("Model performance", expanded=False):
    metrics_path = METRICS_DIR / "results.json"
    if metrics_path.exists():
        all_results = json.loads(metrics_path.read_text())
        best_model = get_model_name()
        if best_model in all_results:
            metrics = all_results[best_model]["metrics"]
            st.caption(f"Test-set metrics for **{best_model}**")
            metric_cols = st.columns(5 if "roc_auc" in metrics else 4)
            for col, label, key, fmt in zip(
                metric_cols,
                ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
                ["accuracy", "precision", "recall", "f1", "roc_auc"],
                [".1%", ".1%", ".1%", ".1%", ".3f"],
            ):
                if key in metrics:
                    col.metric(label, format(metrics[key], fmt))
        else:
            st.caption("Metrics for the active model are not available.")
    else:
        st.caption("No metrics found. Run `python src/train.py` to generate them.")
"""
Thin inference wrapper around the saved pipeline.

Usage:
    from predict import load_pipeline, predict_one
    pipe = load_pipeline()
    result = predict_one(pipe, {
        "age": 44.0, "hypertension": 0, "heart_disease": 0,
        "bmi": 19.31, "HbA1c_level": 6.5, "blood_glucose_level": 200,
        "gender": "Male", "smoking_history": "never",
    })
"""
from pathlib import Path

import joblib
import pandas as pd

from pipeline import ALL_FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pipeline.pkl"


def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained pipeline found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_one(pipe, input_dict: dict) -> dict:
    """Predict diabetes risk for a single set of patient inputs.

    input_dict must contain all of ALL_FEATURES as keys.
    Returns {"prediction": 0 or 1, "label": str, "probability": float}
    """
    missing = [f for f in ALL_FEATURES if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = pd.DataFrame([{f: input_dict[f] for f in ALL_FEATURES}])
    pred = int(pipe.predict(row)[0])
    proba = float(pipe.predict_proba(row)[0][1]) if hasattr(pipe, "predict_proba") else None

    return {
        "prediction": pred,
        "label": "Diabetic" if pred == 1 else "Not diabetic",
        "probability": proba,
    }

"""
Preprocessing definition shared by training and inference.

Keeping this in one place guarantees the exact same encoding/scaling
is applied at train time and at prediction time -- no manual
pd.get_dummies + column-order guessing.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "diabetes"

GENDER_OPTIONS = ["Female", "Male", "Other"]
SMOKING_OPTIONS = ["never", "No Info", "current", "former", "ever", "not current"]


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scale numeric cols, one-hot encode categoricals.

    handle_unknown='ignore' means a category never seen during training
    (e.g. a typo, or a new smoking_history value) won't crash inference --
    it just gets encoded as all-zeros for that feature.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_pipeline(classifier) -> Pipeline:
    """Wrap preprocessing + a classifier into a single fit/predict object."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )

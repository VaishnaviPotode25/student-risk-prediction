from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model.pkl"

FEATURE_LABELS = {
    "attendance": "Attendance",
    "assignment_completion": "Assignment completion",
    "internal_marks": "Internal marks",
    "previous_cgpa": "Previous CGPA",
    "study_hours": "Study hours per day",
    "backlogs": "Backlogs",
}

_bundle = None  # cache: load the model file once, not on every request


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "model.pkl not found. Train it first: python ml/train_model.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _explain(model, features, row):
    """Which features push this student's risk up or down, versus an average student.

    Only works for a scaler + logistic regression pipeline. Other models return no reasons.
    """
    if not hasattr(model, "steps"):
        return [], []
    scaler = model.steps[0][1]
    classifier = model.steps[-1][1]
    if not hasattr(classifier, "coef_"):
        return [], []

    # The scaler centres every feature on the average student (0 = average).
    z = scaler.transform(row)[0]

    # Weight of each feature for "High" compared with "Low".
    classes = list(classifier.classes_)
    weights = classifier.coef_[classes.index("High")] - classifier.coef_[classes.index("Low")]

    effects = weights * z
    items = []
    for feature, effect in zip(features, effects):
        items.append({
            "label": FEATURE_LABELS.get(feature, feature),
            "value": row.iloc[0][feature],
            "effect": float(effect),
        })

    raising = sorted([i for i in items if i["effect"] > 0.1], key=lambda i: -i["effect"])[:3]
    lowering = sorted([i for i in items if i["effect"] < -0.1], key=lambda i: i["effect"])[:3]
    return raising, lowering


def predict_risk(values):
    """values: dict with one number per feature. Returns a dict that is safe to store in MongoDB."""
    bundle = _load_bundle()
    model = bundle["model"]
    features = bundle["features"]

    # Build the row in the exact column order the model was trained with.
    row = pd.DataFrame([{f: float(values[f]) for f in features}], columns=features)

    probabilities = model.predict_proba(row)[0]
    classes = [str(c) for c in model.classes_]
    best = int(probabilities.argmax())

    raising, lowering = _explain(model, features, row)
    for item in raising + lowering:
        item["effect"] = round(item["effect"], 2)
        item["value"] = float(item["value"])

    return {
        "risk_level": classes[best],
        "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(classes, probabilities)},
        "risk_factors": raising,
        "protective_factors": lowering,
        "model_name": str(bundle.get("model_name", "unknown")),
    }
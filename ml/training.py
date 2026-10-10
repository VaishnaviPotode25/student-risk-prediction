import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.common import FEATURES, load_data

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.pkl"
BACKUP_PATH = HERE / "model_previous.pkl"
RESULTS_PATH = HERE / "model_comparison.csv"

MIN_ROWS = 200        # refuse to train on tiny datasets
MIN_PER_CLASS = 20    # every risk level needs enough examples
TOLERANCE = 0.005     # new model may be this much worse (macro F1) and still be accepted


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "SVM": make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def score_model(model, X, y):
    predicted = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "f1_macro": float(f1_score(y, predicted, average="macro")),
        "high_recall": float(recall_score(y, predicted, labels=["High"], average="macro")),
    }


def train_and_select():
    """Train all models, pick the best, and replace model.pkl only if it is not worse."""
    df = load_data()

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} labelled students found; at least {MIN_ROWS} are needed to retrain."
        )
    counts = df["risk_level"].value_counts()
    real_rows = int((df["source"] == "outcome").sum()) if "source" in df else 0
    if len(counts) < 3 or counts.min() < MIN_PER_CLASS:
        raise ValueError(
            f"Each risk level needs at least {MIN_PER_CLASS} examples. Found: {counts.to_dict()}"
        )

    X = df[FEATURES]
    y = df["risk_level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compare every model with 5-fold cross-validation on the training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "high_recall": make_scorer(recall_score, labels=["High"], average="macro"),
    }
    rows = []
    for name, model in build_models().items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "accuracy": float(scores["test_accuracy"].mean()),
            "f1_macro": float(scores["test_f1_macro"].mean()),
            "high_recall": float(scores["test_high_recall"].mean()),
        })
    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    # Train the winner and score it on the held-back students
    best_name = str(comparison.loc[0, "model"])
    best_model = build_models()[best_name]
    best_model.fit(X_train, y_train)
    new_metrics = score_model(best_model, X_test, y_test)

    # Score the model currently in use on the SAME test students
    old_metrics, old_name = None, None
    if MODEL_PATH.exists():
        try:
            old_bundle = joblib.load(MODEL_PATH)
            old_name = str(old_bundle.get("model_name", "unknown"))
            old_metrics = score_model(
                old_bundle["model"], X_test[old_bundle["features"]], y_test
            )
        except Exception:
            old_metrics = None  # unreadable old file: treat it as "no model"

    # The promotion decision
    if old_metrics is None:
        promoted = True
        reason = "No working model existed, so the new model was saved."
    elif new_metrics["f1_macro"] >= old_metrics["f1_macro"] - TOLERANCE:
        promoted = True
        reason = "The new model scored at least as well as the current one, so it replaced it."
    else:
        promoted = False
        reason = "The new model scored worse than the current one, so the current model was kept."

    trained_at = datetime.now(timezone.utc)

    if promoted:
        bundle = {
            "model": best_model,
            "model_name": best_name,
            "features": FEATURES,
            "test_f1_macro": new_metrics["f1_macro"],
            "n_rows": int(len(df)),
            "trained_at": trained_at.isoformat(),
        }
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, BACKUP_PATH)  # keep the previous model
        temp_path = HERE / "model_new.tmp"
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, MODEL_PATH)  # swap in one step
        comparison.round(4).to_csv(RESULTS_PATH, index=False)

    return {
        "trained_at": trained_at,
        "n_rows": int(len(df)),
        "real_rows": real_rows,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "comparison": comparison.round(4).to_dict("records"),
        "best_model": best_name,
        "new_metrics": new_metrics,
        "old_model": old_name,
        "old_metrics": old_metrics,
        "promoted": promoted,
        "reason": reason,
    }
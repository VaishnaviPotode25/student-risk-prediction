from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import FEATURES, RISK_ORDER, load_data

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.pkl"
RESULTS_PATH = HERE / "model_comparison.csv"

df = load_data()
X = df[FEATURES]
y = df["risk_level"]

# The test set is locked away until the very end.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")

MODELS = {
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

# ---- Compare all models with 5-fold cross-validation (training data only) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    # recall for the High class only: how many High students are caught
    "high_recall": make_scorer(recall_score, labels=["High"], average="macro"),
}

rows = []
for name, model in MODELS.items():
    print(f"Evaluating {name}...")
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows.append({
        "model": name,
        "accuracy": scores["test_accuracy"].mean(),
        "f1_macro": scores["test_f1_macro"].mean(),
        "high_recall": scores["test_high_recall"].mean(),
    })

results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
print("\nModel comparison (5-fold cross-validation, training data):")
print(results.round(3).to_string(index=False))
results.round(4).to_csv(RESULTS_PATH, index=False)

# ---- Train the winner and test it ONCE on the held-back students ----
best_name = results.loc[0, "model"]
best_model = MODELS[best_name]
best_model.fit(X_train, y_train)
print(f"\nBest model by macro F1: {best_name}")

y_pred = best_model.predict(X_test)

print("\nPer-class results on the test set:")
print(classification_report(y_test, y_pred, labels=RISK_ORDER))

print("Confusion matrix (rows = actual, columns = predicted):")
matrix = confusion_matrix(y_test, y_pred, labels=RISK_ORDER)
print(pd.DataFrame(
    matrix,
    index=[f"actual {r}" for r in RISK_ORDER],
    columns=[f"pred {r}" for r in RISK_ORDER],
))

# Feature importance exists only for tree-based models
estimator = best_model.steps[-1][1] if hasattr(best_model, "steps") else best_model
if hasattr(estimator, "feature_importances_"):
    print("\nFeature importance:")
    importance = pd.Series(estimator.feature_importances_, index=FEATURES)
    print(importance.sort_values(ascending=False).round(3))

test_f1 = f1_score(y_test, y_pred, average="macro")
joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "test_f1_macro": float(test_f1),
    },
    MODEL_PATH,
)
print(f"\nSaved '{best_name}' to {MODEL_PATH}")
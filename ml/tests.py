import tempfile
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

import joblib
import pandas as pd
from django.test import SimpleTestCase
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml import training
from ml.common import FEATURES
from ml.generate_sample_data import make_student
from ml.predict import MODEL_PATH as REAL_MODEL_PATH
from ml.predict import predict_risk


def sample_frame(n=400):
    """A small labelled dataset made with the same generator as the real sample data."""
    rows = [make_student(i) for i in range(1, n + 1)]
    return pd.DataFrame(rows)[FEATURES + ["risk_level", "source"]]


def fast_models():
    # Two quick models instead of seven, so the tests run in seconds.
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
    }


class TrainingTests(SimpleTestCase):
    """Training runs against a temporary folder, so the real model.pkl is never touched."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        tmp = Path(folder.name)
        self.model_path = tmp / "model.pkl"
        self.backup_path = tmp / "model_previous.pkl"

        patches = [
            patch.object(training, "HERE", tmp),
            patch.object(training, "MODEL_PATH", self.model_path),
            patch.object(training, "BACKUP_PATH", self.backup_path),
            patch.object(training, "RESULTS_PATH", tmp / "model_comparison.csv"),
            patch.object(training, "build_models", fast_models),
            patch.object(training, "load_data", lambda: sample_frame()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_first_training_creates_a_model(self):
        report = training.train_and_select()
        self.assertTrue(report["promoted"])
        self.assertIn("No working model", report["reason"])
        self.assertTrue(self.model_path.exists())

    def test_retraining_on_same_data_replaces_and_keeps_a_backup(self):
        training.train_and_select()
        report = training.train_and_select()
        self.assertTrue(report["promoted"])
        self.assertTrue(self.backup_path.exists())

    def test_worse_model_is_not_promoted(self):
        training.train_and_select()
        before = self.model_path.read_bytes()
        # A tolerance of -1 demands an impossible improvement.
        with patch.object(training, "TOLERANCE", -1):
            report = training.train_and_select()
        self.assertFalse(report["promoted"])
        self.assertEqual(self.model_path.read_bytes(), before)  # file unchanged

    def test_better_model_replaces_a_useless_one(self):
        df = sample_frame()
        useless = DummyClassifier(strategy="most_frequent").fit(df[FEATURES], df["risk_level"])
        joblib.dump(
            {"model": useless, "model_name": "Dummy", "features": FEATURES}, self.model_path
        )
        report = training.train_and_select()
        self.assertTrue(report["promoted"])
        self.assertGreater(
            report["new_metrics"]["f1_macro"], report["old_metrics"]["f1_macro"]
        )

    def test_tiny_dataset_is_refused(self):
        with patch.object(training, "load_data", lambda: sample_frame(50)):
            with self.assertRaises(ValueError):
                training.train_and_select()


class RealModelTests(SimpleTestCase):
    @skipUnless(REAL_MODEL_PATH.exists(), "Train a model first: python -m ml.train_model")
    def test_clear_cases_are_classified_correctly(self):
        high = predict_risk({
            "attendance": 40, "assignment_completion": 30, "internal_marks": 35,
            "previous_cgpa": 4.0, "study_hours": 1, "backlogs": 4,
        })
        low = predict_risk({
            "attendance": 95, "assignment_completion": 95, "internal_marks": 90,
            "previous_cgpa": 9.0, "study_hours": 5, "backlogs": 0,
        })
        self.assertEqual(high["risk_level"], "High")
        self.assertEqual(low["risk_level"], "Low")
        self.assertAlmostEqual(sum(high["probabilities"].values()), 100, delta=0.5)
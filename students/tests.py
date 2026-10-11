from unittest.mock import patch

from bson import ObjectId
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from .mongo import (
    get_client,
    get_predictions_collection,
    get_students_collection,
    get_training_runs_collection,
)

User = get_user_model()
TEST_DB = "student_prediction_test"

VALID_STUDENT = {
    "name": "Test Student",
    "attendance": 50,
    "assignment_completion": 40,
    "internal_marks": 45,
    "previous_cgpa": 5.0,
    "study_hours": 2,
    "backlogs": 2,
    "actual_outcome": "",
}

PREDICT_INPUT = {
    "attendance": 45,
    "assignment_completion": 35,
    "internal_marks": 40,
    "previous_cgpa": 4.5,
    "study_hours": 1,
    "backlogs": 3,
}

FAKE_PREDICTION = {
    "risk_level": "High",
    "probabilities": {"Low": 1.0, "Medium": 2.0, "High": 97.0},
    "risk_factors": [],
    "protective_factors": [],
    "model_name": "Fake model",
}


def fake_report(promoted=True):
    metrics = {"accuracy": 0.9, "f1_macro": 0.9, "high_recall": 0.9}
    return {
        "trained_at": timezone.now(),
        "n_rows": 2000,
        "class_counts": {},
        "comparison": [],
        "best_model": "Logistic Regression",
        "new_metrics": metrics,
        "old_model": "Logistic Regression",
        "old_metrics": metrics,
        "promoted": promoted,
        "reason": "test reason",
    }


@override_settings(MONGO_DB_NAME=TEST_DB)
class MongoTestCase(TestCase):
    """Every test runs against a throwaway MongoDB database, never the real one."""

    def setUp(self):
        self._drop_test_db()
        self.addCleanup(self._drop_test_db)

    def _drop_test_db(self):
        # Safety check: refuse to drop anything except the test database.
        self.assertEqual(settings.MONGO_DB_NAME, TEST_DB)
        get_client().drop_database(TEST_DB)

    def add_student_doc(self, **extra):
        doc = {"name": "Test Student", "attendance": 50, "created_at": timezone.now(), **extra}
        return get_students_collection().insert_one(doc).inserted_id


class LoginProtectionTests(MongoTestCase):
    PROTECTED = ["/dashboard/", "/students/", "/students/add/", "/predict/"]

    def test_home_page_is_public(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_protected_pages_redirect_to_login(self):
        for url in self.PROTECTED:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertRedirects(
                    response, f"/accounts/login/?next={url}", fetch_redirect_response=False
                )

    def test_logged_in_user_can_open_pages(self):
        self.client.force_login(User.objects.create_user("viewer", password="pw12345!"))
        for url in self.PROTECTED:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_anonymous_user_cannot_delete(self):
        student_id = self.add_student_doc()
        self.client.post(f"/students/{student_id}/delete/")
        self.assertEqual(get_students_collection().count_documents({}), 1)


class StudentCrudTests(MongoTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(User.objects.create_user("staffer", password="pw12345!"))

    def test_add_student_without_outcome_has_no_label(self):
        self.client.post("/students/add/", VALID_STUDENT)
        doc = get_students_collection().find_one({"name": "Test Student"})
        self.assertIsNotNone(doc)
        self.assertNotIn("risk_level", doc)
        self.assertNotIn("source", doc)

    def test_add_student_with_outcome_gets_label(self):
        data = {**VALID_STUDENT, "actual_outcome": "Failed or dropped out"}
        self.client.post("/students/add/", data)
        doc = get_students_collection().find_one({"name": "Test Student"})
        self.assertEqual(doc["risk_level"], "High")
        self.assertEqual(doc["source"], "outcome")
        self.assertEqual(doc["actual_outcome"], "Failed or dropped out")

    def test_invalid_student_is_rejected(self):
        data = {**VALID_STUDENT, "attendance": 150}
        response = self.client.post("/students/add/", data)
        self.assertEqual(response.status_code, 200)  # form shown again
        self.assertIn("attendance", response.context["form"].errors)
        self.assertEqual(get_students_collection().count_documents({}), 0)

    def test_clearing_an_outcome_removes_its_label(self):
        data = {**VALID_STUDENT, "actual_outcome": "Passed all subjects"}
        self.client.post("/students/add/", data)
        student_id = get_students_collection().find_one()["_id"]

        self.client.post(f"/students/{student_id}/edit/", VALID_STUDENT)  # outcome blank
        doc = get_students_collection().find_one({"_id": student_id})
        for field in ("actual_outcome", "risk_level", "source"):
            self.assertNotIn(field, doc)

    def test_editing_a_synthetic_student_keeps_its_label(self):
        student_id = self.add_student_doc(risk_level="Low", source="synthetic")
        self.client.post(f"/students/{student_id}/edit/", VALID_STUDENT)  # outcome blank
        doc = get_students_collection().find_one({"_id": student_id})
        self.assertEqual(doc["risk_level"], "Low")
        self.assertEqual(doc["source"], "synthetic")

    def test_delete_requires_post(self):
        student_id = self.add_student_doc()
        self.client.get(f"/students/{student_id}/delete/")
        self.assertEqual(get_students_collection().count_documents({}), 1)

    def test_delete_with_post_removes_student(self):
        student_id = self.add_student_doc()
        self.client.post(f"/students/{student_id}/delete/")
        self.assertEqual(get_students_collection().count_documents({}), 0)

    def test_unknown_student_gives_404(self):
        response = self.client.get(f"/students/{ObjectId()}/edit/")
        self.assertEqual(response.status_code, 404)


class PredictionTests(MongoTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(User.objects.create_user("predictor", password="pw12345!"))

    @patch("students.views.predict_risk", return_value=FAKE_PREDICTION)
    def test_valid_prediction_is_saved(self, mock_predict):
        response = self.client.post("/predict/", PREDICT_INPUT)
        saved = get_predictions_collection().find_one()
        self.assertIsNotNone(saved)
        self.assertRedirects(
            response, f"/predict/{saved['_id']}/", fetch_redirect_response=False
        )
        self.assertEqual(saved["risk_level"], "High")
        self.assertEqual(saved["student_name"], "Unnamed student")
        mock_predict.assert_called_once()

    @patch("students.views.predict_risk", return_value=FAKE_PREDICTION)
    def test_result_page_shows_the_risk_level(self, _mock):
        self.client.post("/predict/", PREDICT_INPUT)
        saved = get_predictions_collection().find_one()
        response = self.client.get(f"/predict/{saved['_id']}/")
        self.assertContains(response, "High")

    @patch("students.views.predict_risk")
    def test_invalid_input_never_reaches_the_model(self, mock_predict):
        response = self.client.post("/predict/", {**PREDICT_INPUT, "attendance": 150})
        self.assertEqual(response.status_code, 200)
        mock_predict.assert_not_called()
        self.assertEqual(get_predictions_collection().count_documents({}), 0)


class RetrainPermissionTests(MongoTestCase):
    def setUp(self):
        super().setUp()
        self.staff = User.objects.create_user("staff_user", password="pw12345!", is_staff=True)
        self.regular = User.objects.create_user("regular_user", password="pw12345!")

    @patch("students.views.train_and_select")
    def test_anonymous_cannot_retrain(self, mock_train):
        response = self.client.post("/retrain/")
        self.assertIn("/accounts/login/", response.url)
        mock_train.assert_not_called()

    @patch("students.views.train_and_select")
    def test_non_staff_cannot_retrain(self, mock_train):
        self.client.force_login(self.regular)
        response = self.client.post("/retrain/")
        self.assertIn("/accounts/login/", response.url)
        mock_train.assert_not_called()

    @patch("students.views.train_and_select")
    def test_get_request_does_not_retrain(self, mock_train):
        self.client.force_login(self.staff)
        response = self.client.get("/retrain/")
        self.assertRedirects(response, "/dashboard/", fetch_redirect_response=False)
        mock_train.assert_not_called()

    @patch("students.views.train_and_select", return_value=fake_report())
    def test_staff_post_retrains_and_logs_the_run(self, mock_train):
        self.client.force_login(self.staff)
        response = self.client.post("/retrain/")
        self.assertRedirects(response, "/dashboard/", fetch_redirect_response=False)
        mock_train.assert_called_once()
        run = get_training_runs_collection().find_one()
        self.assertEqual(run["triggered_by"], "staff_user")
        self.assertTrue(run["promoted"])

    @patch("students.views.train_and_select", side_effect=ValueError("Not enough data"))
    def test_training_error_is_shown_not_crashed(self, _mock):
        self.client.force_login(self.staff)
        response = self.client.post("/retrain/", follow=True)
        self.assertContains(response, "Not enough data")
        self.assertEqual(get_training_runs_collection().count_documents({}), 0)


class StudentListTests(MongoTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(User.objects.create_user("lister", password="pw12345!"))

    def test_search_filters_by_name(self):
        self.add_student_doc(name="Asha Rao")
        self.add_student_doc(name="Ben Cole")
        response = self.client.get("/students/?q=asha")
        self.assertContains(response, "Asha Rao")
        self.assertNotContains(response, "Ben Cole")

    def test_search_text_is_matched_literally(self):
        self.add_student_doc(name="A.B")
        self.add_student_doc(name="AXB")
        response = self.client.get("/students/?q=A.B")
        self.assertEqual(response.context["total"], 1)

    def test_risk_filter(self):
        self.add_student_doc(name="Risky", risk_level="High", source="synthetic")
        self.add_student_doc(name="Steady", risk_level="Low", source="synthetic")
        response = self.client.get("/students/?risk=High")
        self.assertContains(response, "Risky")
        self.assertNotContains(response, "Steady")

    def test_pagination(self):
        for i in range(30):
            self.add_student_doc(name=f"Student {i:02d}")
        first = self.client.get("/students/")
        self.assertEqual(first.context["total"], 30)
        self.assertEqual(first.context["pages"], 2)
        self.assertEqual(len(first.context["students"]), 25)
        second = self.client.get("/students/?page=2")
        self.assertEqual(len(second.context["students"]), 5)

    def test_bad_page_number_does_not_crash(self):
        response = self.client.get("/students/?page=abc")
        self.assertEqual(response.status_code, 200)
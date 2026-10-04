from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("students/", views.student_list, name="student_list"),
    path("students/add/", views.add_student, name="add_student"),
    path("students/<str:student_id>/edit/", views.edit_student, name="edit_student"),
    path("students/<str:student_id>/delete/", views.delete_student, name="delete_student"),
    path("predict/", views.predict_student, name="predict_student"),
    path("predict/<str:prediction_id>/", views.prediction_detail, name="prediction_detail"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("retrain/", views.retrain_model, name="retrain_model"),
]
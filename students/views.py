from django.shortcuts import render
from .mongo import get_students_collection


def home(request):
    students = get_students_collection()

    context = {
        "project_name": "Smart Student Performance Analytics and Risk Prediction System",
        "total_students": students.count_documents({}),
        "features": [
            "Store student data in MongoDB",
            "Train a machine learning model",
            "Predict student risk level",
            "View analytics on a dashboard",
        ],
    }
    return render(request, "students/home.html", context)
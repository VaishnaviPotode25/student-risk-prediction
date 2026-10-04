from django.shortcuts import render


def home(request):
    context = {
        "project_name": "Smart Student Performance Analytics and Risk Prediction System",
        "total_students": 0,
        "features": [
            "Store student data in MongoDB",
            "Train a machine learning model",
            "Predict student risk level",
            "View analytics on a dashboard",
        ],
    }
    return render(request, "students/home.html", context)
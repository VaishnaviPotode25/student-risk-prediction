from datetime import datetime

from django.shortcuts import render, redirect

from .forms import StudentForm
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


def student_list(request):
    students = list(get_students_collection().find().sort("created_at", -1))
    return render(request, "students/student_list.html", {"students": students})


def add_student(request):
    if request.method == "POST":
        form = StudentForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            data["created_at"] = datetime.now()
            get_students_collection().insert_one(data)
            return redirect("student_list")
    else:
        form = StudentForm()

    return render(request, "students/add_student.html", {"form": form})
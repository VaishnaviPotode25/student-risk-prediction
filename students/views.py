from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from django.http import Http404
from django.shortcuts import render, redirect

from .forms import StudentForm
from .mongo import get_students_collection


def get_student_or_404(student_id):
    """Fetch one student by its string id, or raise a 404 page."""
    try:
        student = get_students_collection().find_one({"_id": ObjectId(student_id)})
    except InvalidId:
        raise Http404("Invalid student id")
    if student is None:
        raise Http404("Student not found")
    return student


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
    # Templates can't use a name starting with "_", so we add a plain "id" field.
    for s in students:
        s["id"] = str(s["_id"])
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

    return render(request, "students/add_student.html", {"form": form, "title": "Add Student"})


def edit_student(request, student_id):
    student = get_student_or_404(student_id)

    if request.method == "POST":
        form = StudentForm(request.POST)
        if form.is_valid():
            get_students_collection().update_one(
                {"_id": student["_id"]},
                {"$set": form.cleaned_data},
            )
            return redirect("student_list")
    else:
        form = StudentForm(initial=student)

    return render(request, "students/add_student.html", {"form": form, "title": "Edit Student"})


def delete_student(request, student_id):
    student = get_student_or_404(student_id)

    # Deleting must only happen on POST, never from a plain link click.
    if request.method == "POST":
        get_students_collection().delete_one({"_id": student["_id"]})

    return redirect("student_list")
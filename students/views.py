from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from django.http import Http404
from django.shortcuts import render, redirect

from ml.predict import predict_risk

from .forms import FEATURE_FIELDS, PredictionForm, StudentForm
from .mongo import get_predictions_collection, get_students_collection

from django.contrib.auth.decorators import login_required


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


@login_required
def student_list(request):
    students = list(get_students_collection().find().sort("created_at", -1))
    # Templates can't use a name starting with "_", so we add a plain "id" field.
    for s in students:
        s["id"] = str(s["_id"])
    return render(request, "students/student_list.html", {"students": students})


@login_required
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


@login_required
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


@login_required
def delete_student(request, student_id):
    student = get_student_or_404(student_id)

    # Deleting must only happen on POST, never from a plain link click.
    if request.method == "POST":
        get_students_collection().delete_one({"_id": student["_id"]})

    return redirect("student_list")


@login_required
def predict_student(request):
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            inputs = {field: data[field] for field in FEATURE_FIELDS}
            result = predict_risk(inputs)

            record = {
                "student_name": data["name"] or "Unnamed student",
                "inputs": inputs,
                **result,
                "predicted_at": datetime.now(),
            }
            inserted = get_predictions_collection().insert_one(record)
            return redirect("prediction_detail", prediction_id=str(inserted.inserted_id))
    else:
        form = PredictionForm()

    return render(request, "students/predict.html", {"form": form})


@login_required
def prediction_detail(request, prediction_id):
    try:
        prediction = get_predictions_collection().find_one({"_id": ObjectId(prediction_id)})
    except InvalidId:
        raise Http404("Invalid prediction id")
    if prediction is None:
        raise Http404("Prediction not found")

    return render(request, "students/prediction_detail.html", {"p": prediction})


@login_required
def dashboard(request):
    students = get_students_collection()
    predictions = get_predictions_collection()

    # MongoDB groups the labelled students by risk level and computes the averages.
    pipeline = [
        {"$match": {"risk_level": {"$exists": True}}},
        {"$group": {
            "_id": "$risk_level",
            "count": {"$sum": 1},
            "avg_attendance": {"$avg": "$attendance"},
            "avg_assignments": {"$avg": "$assignment_completion"},
            "avg_marks": {"$avg": "$internal_marks"},
            "avg_cgpa": {"$avg": "$previous_cgpa"},
            "avg_backlogs": {"$avg": "$backlogs"},
        }},
    ]
    grouped = {row["_id"]: row for row in students.aggregate(pipeline)}

    summary = []
    for level in ["Low", "Medium", "High"]:
        row = grouped.get(level, {})
        summary.append({
            "level": level,
            "count": row.get("count", 0),
            "avg_attendance": round(row.get("avg_attendance", 0), 1),
            "avg_assignments": round(row.get("avg_assignments", 0), 1),
            "avg_marks": round(row.get("avg_marks", 0), 1),
            "avg_cgpa": round(row.get("avg_cgpa", 0), 2),
            "avg_backlogs": round(row.get("avg_backlogs", 0), 2),
        })

    recent = list(predictions.find().sort("predicted_at", -1).limit(10))
    for p in recent:
        p["id"] = str(p["_id"])

    context = {
        "total_students": students.count_documents({}),
        "labelled_students": sum(row["count"] for row in summary),
        "total_predictions": predictions.count_documents({}),
        "summary": summary,
        "recent_predictions": recent,
        # json_script in the template passes this safely to JavaScript
        "chart_data": {
            "levels": [row["level"] for row in summary],
            "counts": [row["count"] for row in summary],
            "avg_attendance": [row["avg_attendance"] for row in summary],
            "avg_assignments": [row["avg_assignments"] for row in summary],
            "avg_marks": [row["avg_marks"] for row in summary],
        },
    }
    return render(request, "students/dashboard.html", context)
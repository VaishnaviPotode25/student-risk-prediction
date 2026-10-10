from django import forms

FEATURE_FIELDS = [
    "attendance",
    "assignment_completion",
    "internal_marks",
    "previous_cgpa",
    "study_hours",
    "backlogs",
]

OUTCOME_TO_RISK = {
    "Passed all subjects": "Low",
    "Passed with new backlog(s)": "Medium",
    "Failed or dropped out": "High",
}

OUTCOME_CHOICES = [("", "Not known yet")] + [(o, o) for o in OUTCOME_TO_RISK]


class StudentForm(forms.Form):
    name = forms.CharField(max_length=100)
    attendance = forms.IntegerField(min_value=0, max_value=100, label="Attendance (%)")
    assignment_completion = forms.IntegerField(min_value=0, max_value=100, label="Assignment completion (%)")
    internal_marks = forms.IntegerField(min_value=0, max_value=100)
    previous_cgpa = forms.FloatField(min_value=0, max_value=10, label="Previous CGPA")
    study_hours = forms.IntegerField(min_value=0, max_value=24, label="Study hours per day")
    backlogs = forms.IntegerField(min_value=0)
    actual_outcome = forms.ChoiceField(
        choices=OUTCOME_CHOICES,
        required=False,
        label="Actual outcome (fill in at the end of the semester)",
    )


class PredictionForm(StudentForm):
    # Same fields as StudentForm, but the name is optional for a quick prediction.
    name = forms.CharField(max_length=100, required=False, label="Student name (optional)")
    actual_outcome = None  # a prediction has no outcome yet, so remove the field
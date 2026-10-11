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
    attendance = forms.IntegerField(
        min_value=0, max_value=100, label="Attendance (%)",
        help_text="Share of classes attended, from 0 to 100.",
    )
    assignment_completion = forms.IntegerField(
        min_value=0, max_value=100, label="Assignment completion (%)",
        help_text="Share of assignments handed in, from 0 to 100.",
    )
    internal_marks = forms.IntegerField(
        min_value=0, max_value=100,
        help_text="Internal assessment score out of 100.",
    )
    previous_cgpa = forms.FloatField(
        min_value=0, max_value=10, label="Previous CGPA",
        help_text="CGPA from the last semester, from 0 to 10.",
    )
    study_hours = forms.IntegerField(
        min_value=0, max_value=24, label="Study hours per day",
        help_text="Average hours of self-study per day.",
    )
    backlogs = forms.IntegerField(
        min_value=0,
        help_text="Number of subjects not yet cleared.",
    )
    actual_outcome = forms.ChoiceField(
        choices=OUTCOME_CHOICES,
        required=False,
        label="Actual outcome",
        help_text="Leave as 'Not known yet' until results are out. A recorded outcome becomes a training label.",
    )


class PredictionForm(StudentForm):
    # Same fields as StudentForm, but the name is optional for a quick prediction.
    name = forms.CharField(max_length=100, required=False, label="Student name (optional)")
    actual_outcome = None  # a prediction has no outcome yet, so remove the field
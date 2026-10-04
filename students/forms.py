from django import forms


class StudentForm(forms.Form):
    name = forms.CharField(max_length=100)
    attendance = forms.IntegerField(min_value=0, max_value=100, label="Attendance (%)")
    assignment_completion = forms.IntegerField(min_value=0, max_value=100, label="Assignment completion (%)")
    internal_marks = forms.IntegerField(min_value=0, max_value=100)
    previous_cgpa = forms.FloatField(min_value=0, max_value=10, label="Previous CGPA")
    study_hours = forms.IntegerField(min_value=0, max_value=24, label="Study hours per day")
    backlogs = forms.IntegerField(min_value=0)
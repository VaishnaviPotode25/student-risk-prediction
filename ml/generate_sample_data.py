import random
from datetime import datetime, timezone

from pymongo import MongoClient

from ml.common import DB_NAME, MONGO_URI
NUM_STUDENTS = 2000

random.seed(42)  # same seed = same data every run (reproducible)


def clip(value, low, high):
    return max(low, min(high, value))


def make_student(number):
    # One hidden "diligence" value makes the features move together,
    # like real life: students with good attendance tend to have better marks.
    d = random.gauss(0, 1)

    attendance = round(clip(75 + 12 * d + random.gauss(0, 6), 30, 100))
    assignment = round(clip(72 + 14 * d + random.gauss(0, 8), 10, 100))
    marks = round(clip(62 + 14 * d + random.gauss(0, 8), 10, 100))
    cgpa = round(clip(6.8 + 1.1 * d + random.gauss(0, 0.4), 3.0, 10.0), 2)
    study_hours = int(round(clip(3 + 1.2 * d + random.gauss(0, 0.8), 0, 10)))
    backlogs = int(round(clip(-0.9 * d + random.gauss(0.2, 0.8), 0, 8)))

    # Risk score: higher = more at risk. Noise keeps it from being a perfect formula.
    score = (
        (100 - attendance) * 0.30
        + (100 - assignment) * 0.20
        + (100 - marks) * 0.20
        + (10 - cgpa) * 5
        + backlogs * 4
        + (5 - study_hours) * 1.5
        + random.gauss(0, 3)
    )

    if score < 40:
        risk = "Low"
    elif score < 58:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "name": f"Student {number:04d}",
        "attendance": attendance,
        "assignment_completion": assignment,
        "internal_marks": marks,
        "previous_cgpa": cgpa,
        "study_hours": study_hours,
        "backlogs": backlogs,
        "risk_level": risk,
        "source": "synthetic",
        "created_at": datetime.now(timezone.utc),
    }


def main():
    students = MongoClient(MONGO_URI)[DB_NAME]["students"]

    # Remove old synthetic rows so running the script twice doesn't double the data.
    # Real students (added through the website) have no "source" field and are kept.
    deleted = students.delete_many({"source": "synthetic"}).deleted_count
    print(f"Removed {deleted} old synthetic students")

    docs = [make_student(i) for i in range(1, NUM_STUDENTS + 1)]
    students.insert_many(docs)
    print(f"Inserted {len(docs)} synthetic students")

    for level in ("Low", "Medium", "High"):
        count = students.count_documents({"source": "synthetic", "risk_level": level})
        print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
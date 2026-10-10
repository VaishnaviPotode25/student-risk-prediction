import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "student_prediction_db"

FEATURES = [
    "attendance",
    "assignment_completion",
    "internal_marks",
    "previous_cgpa",
    "study_hours",
    "backlogs",
]
RISK_ORDER = ["Low", "Medium", "High"]


def load_data():
    """Pull labelled students out of MongoDB into a pandas DataFrame."""
    collection = MongoClient(MONGO_URI)[DB_NAME]["students"]

    query = {"risk_level": {"$exists": True}}
    projection = {"_id": 0, "risk_level": 1}
    projection["source"] = 1
    projection.update({feature: 1 for feature in FEATURES})

    return pd.DataFrame(list(collection.find(query, projection)))
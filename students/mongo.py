from django.conf import settings
from pymongo import MongoClient

# tz_aware=True makes MongoDB return timezone-aware datetimes, so Django can
# show them in the local time zone instead of raw UTC.
_client = MongoClient(settings.MONGO_URI, tz_aware=True)
_db = _client[settings.MONGO_DB_NAME]


def get_students_collection():
    return _db["students"]


def get_predictions_collection():
    return _db["predictions"]


def get_training_runs_collection():
    return _db["training_runs"]
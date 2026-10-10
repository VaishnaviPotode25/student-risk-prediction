from django.conf import settings
from pymongo import MongoClient

_client = None


def get_client():
    """One MongoClient for the whole app, created on first use."""
    global _client
    if _client is None:
        # tz_aware=True makes MongoDB return timezone-aware datetimes.
        _client = MongoClient(settings.MONGO_URI, tz_aware=True)
    return _client


def get_db():
    # The database name is read on every call, so tests can point at a test database.
    return get_client()[settings.MONGO_DB_NAME]


def get_students_collection():
    return get_db()["students"]


def get_predictions_collection():
    return get_db()["predictions"]


def get_training_runs_collection():
    return get_db()["training_runs"]
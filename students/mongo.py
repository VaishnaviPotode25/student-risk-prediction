from django.conf import settings
from pymongo import MongoClient

# One client for the whole app. MongoClient manages its own connection pool,
# so we create it once and reuse it.
_client = MongoClient(settings.MONGO_URI)
_db = _client[settings.MONGO_DB_NAME]


def get_students_collection():
    return _db["students"]


def get_predictions_collection():
    return _db["predictions"]
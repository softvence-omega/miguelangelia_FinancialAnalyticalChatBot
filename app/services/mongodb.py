import json
from bson import json_util
from pymongo import MongoClient
from app.core.config import settings

client = MongoClient(settings.db_uri)
db = client["miguelangelia"]
collection = db["chat-histories"]

for doc in collection.find({"thread_id": "65f0a3a5-6a3a-47e6-853e-b50e1d1ccb34"}):
    print(json.dumps(doc, indent=4, default=json_util.default))




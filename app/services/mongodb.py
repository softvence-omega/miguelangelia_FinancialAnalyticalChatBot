from pymongo import MongoClient
from app.core.config import settings
# Get MongoDB URL
db_url = settings.db_uri

# Debug check
print("DB_URL:", db_url)

# Connect to MongoDB Atlas
client = MongoClient(db_url)
db = client["miguelangelia"]
db.create_collection("chat-histories")
print(db.list_collection_names())


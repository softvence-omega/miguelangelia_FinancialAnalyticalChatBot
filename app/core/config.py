

from pydantic_settings import BaseSettings
from pymongo import MongoClient

class Settings(BaseSettings):
    openai_api_key: str
    db_uri: str
    db_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# MongoDB connection
client = MongoClient(settings.db_uri)
db = client[settings.db_name]
history_collection = db["chat-histories"]
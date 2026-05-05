from google import genai
from api.app.database.manager import DatabaseManager
from api.app.config import api_key

def get_db():
    db = DatabaseManager()
    try:
        yield db
    finally:
        db.close_all()

def get_genai_client():
    return genai.Client(api_key=api_key)
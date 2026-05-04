from google import genai
from .config import api_key
from app.config import DB_HOST, DB_NAME, db_user, db_pw
import psycopg2

def get_db():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=db_user, password=db_pw)
    try:
        yield conn
    finally:
        conn.close()

def get_genai_client():
    return genai.Client(api_key=api_key)
import json
import pandas as pd
from random import randint
from datetime import datetime

from celery_app import celery
from google import genai
from database.manager import DatabaseManager
from utils.db_utils import save_resume_data
from utils.embeddings import df_with_embedding_column, embed_texts
from services.resume_service import extract_professional_structured_data
from config import api_key


def _get_redis():
    """Return a plain redis client for status tracking."""
    import redis
    return redis.Redis(host="localhost", port=6379, db=1, decode_responses=True)


def set_status(upload_id: str, status: str, detail: dict | None = None):
    r = _get_redis()
    payload = {"status": status, **(detail or {})}
    r.set(f"resume:status:{upload_id}", json.dumps(payload), ex=86400)  # TTL 24 h


def get_status(upload_id: str) -> dict | None:
    r = _get_redis()
    raw = r.get(f"resume:status:{upload_id}")
    return json.loads(raw) if raw else None


@celery.task(bind=True, max_retries=3, default_retry_delay=10)
def process_resume_task(self, upload_id: str, cv_md_text: str):
    """
    Heavy processing extracted from process_resume_upload.
    Receives already-parsed markdown text so the UploadFile handle
    doesn't need to be kept open.
    """
    set_status(upload_id, "processing")

    try:
        profile_obj = extract_professional_structured_data(cv_md_text)
        genai_client = genai.Client(api_key=api_key)
        position_embedding = embed_texts(genai_client, [profile_obj["position"]["name"]])[0]

        login_example = f"person{randint(0, 9999)}@mail.com"
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv_obj = {
            "id": upload_id,
            "user_id": login_example,
            "upload_id": upload_id,
            "upload_date": upload_date,
            "description": cv_md_text,
            "industries": profile_obj["industries"],
            "position": profile_obj["position"]["name"],
            "time_experience_months": profile_obj["position"]["time_experience_months"],
            "position_embedding": position_embedding,
        }

        cv_df = pd.DataFrame([cv_obj])
        hard_skills_df = pd.DataFrame(profile_obj["hard_skills"])
        soft_skills_df = pd.DataFrame(profile_obj["soft_skills"])

        hard_skills_df = df_with_embedding_column(genai_client, hard_skills_df, "description")
        soft_skills_df = df_with_embedding_column(genai_client, soft_skills_df, "description")

        hard_skills_df["resume_id"] = upload_id
        soft_skills_df["resume_id"] = upload_id

        hard_skills_df = hard_skills_df[["resume_id", "description", "weight", "time_experience_months", "embedding"]]
        soft_skills_df = soft_skills_df[["resume_id", "description", "weight", "embedding"]]

        db = DatabaseManager()
        with db.get_conn() as conn:
            save_resume_data(
                conn=conn,
                cv_df=cv_df,
                hard_skills_df=hard_skills_df,
                soft_skills_df=soft_skills_df,
            )

        set_status(upload_id, "done", {
            "resume_id": upload_id,
            "user_id": login_example,
            "upload_date": upload_date,
            "position": profile_obj["position"]["name"],
            "industries": profile_obj["industries"],
            "time_experience_months": profile_obj["position"]["time_experience_months"],
            "hard_skills": profile_obj["hard_skills"],
            "soft_skills": profile_obj["soft_skills"],
        })

    except Exception as exc:
        set_status(upload_id, "failed", {"error": str(exc)})
        raise self.retry(exc=exc)

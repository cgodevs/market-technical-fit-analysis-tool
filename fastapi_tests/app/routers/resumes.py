import pymupdf

from .. utils import *
from fastapi import UploadFile, status, HTTPException, APIRouter
from random import randint
from datetime import datetime
import pymupdf4llm
import uuid

router = APIRouter()


@router.post("/resumes/upload", status_code=status.HTTP_201_CREATED)
async def send_resume(file: UploadFile):

    login_example = f"person{randint(0, 9999)}@mail.com"
    resume_upload_id = str(uuid.uuid4())

    client = genai.Client(api_key=api_key)

    file_bytes = await file.read()
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    cv_md_text = pymupdf4llm.to_markdown(doc)
    profile_obj = extract_professional_structured_data(cv_md_text)
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv_obj = {
        "id": resume_upload_id,
        "user_id": login_example,
        "upload_id": resume_upload_id,
        "upload_date": upload_date,
        "description": cv_md_text,
        "industries": profile_obj["industries"],
        "position": profile_obj["position"]["name"],
        "time_experience_months": profile_obj["position"]["time_experience_months"],
        "position_embedding": embed_texts(client, [profile_obj["position"]["name"]])[0]
    }

    cv_df = pd.DataFrame([cv_obj])
    hard_skills_df = pd.DataFrame(profile_obj["hard_skills"])
    soft_skills_df = pd.DataFrame(profile_obj["soft_skills"])

    hard_skills_df = recreate_df_with_embeddings(client, hard_skills_df, "description")
    soft_skills_df = recreate_df_with_embeddings(client, soft_skills_df, "description")

    hard_skills_df["resume_id"] = resume_upload_id
    soft_skills_df["resume_id"] = resume_upload_id

    hard_skills_df = hard_skills_df[["resume_id", "description", "weight", "time_experience_months", "embedding"]]
    soft_skills_df = soft_skills_df[["resume_id", "description", "weight", "embedding"]]

    save_df_to_database(df=cv_df, table_name=RESUMES_TABLE)
    save_df_to_database(df=hard_skills_df, table_name=CANDIDATE_HARD_SKILLS_TABLE)
    save_df_to_database(df=soft_skills_df, table_name=CANDIDATE_SOFT_SKILLS_TABLE)

    return {
        "message": "SUCCESS",
        "upload_id": resume_upload_id,
        "resume_id": resume_upload_id,
        "user_id": login_example,
        "upload_date": upload_date,
        "position": profile_obj["position"]["name"],
        "industries": profile_obj["industries"],
        "time_experience_months": profile_obj["position"]["time_experience_months"],
        "hard_skills": profile_obj["hard_skills"],
        "soft_skills": profile_obj["soft_skills"]
    }


@router.get("/resumes/{resume_id}", status_code=status.HTTP_200_OK)
async def candidate_resume(resume_id: str):
    db = DatabaseManager()
    try:
        resume_df = db.get_resume(resume_id).drop(columns=["position_embedding"])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch resume") from e
    finally:
        db.close_all()

    resume_obj = resume_df.to_dict(orient="records")[0] if not resume_df.empty else {}
    if resume_obj == {}:
        raise HTTPException(status_code=404, detail=f"Resume with id {resume_id} not found")

    return resume_obj
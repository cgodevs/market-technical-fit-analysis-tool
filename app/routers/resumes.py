from ..services.resume_service import *
from ..utils.embeddings import embed_texts
from fastapi import status, HTTPException, APIRouter, UploadFile
from datetime import datetime
from random import randint
import uuid

router = APIRouter()


@router.post("/resumes/upload", status_code=status.HTTP_201_CREATED)
async def send_resume(file: UploadFile):

    login_example = f"person{randint(0, 9999)}@mail.com"
    resume_upload_id = str(uuid.uuid4())

    cv_md_text = await parse_resume(file)
    if not cv_md_text:
        raise HTTPException(status_code=500, detail="Failed to parse resume") 
    
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
        "position_embedding": embed_texts([profile_obj["position"]["name"]])[0]
    }
    save_resume_to_database(cv_obj)
    save_skills_to_database(profile_obj, resume_upload_id)

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
async def get_resume(resume_id: str):
    resume_obj = get_resume_obj(resume_id)
    if not resume_obj:
        raise HTTPException(status_code=404, detail=f"Resume with id {resume_id} not found")
    return resume_obj


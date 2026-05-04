from fastapi import Depends, status, HTTPException, APIRouter, UploadFile
from services.resume_service import *
from dependencies import get_db, get_genai_client
from exceptions import ResumeParsingError, StructuredOutputParsingError, EmbeddingError, ResumeNotFoundError

router = APIRouter()


@router.post("/resumes/upload", status_code=status.HTTP_201_CREATED)
async def send_resume(
    file: UploadFile,
    db: DatabaseManager = Depends(get_db),
    client: genai.Client = Depends(get_genai_client),
):
    try:
        upload_result = await process_resume_upload(client, db, file)
    except ResumeParsingError as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except StructuredOutputParsingError as e:
        raise HTTPException(status_code=500, detail=e.detail)
    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=e.detail)
    except ResumeProcessingError as e:
        raise HTTPException(status_code=500, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

    return {
        "message": "SUCCESS",
        "upload_id": upload_result["resume_upload_id"],
        "resume_id": upload_result["resume_upload_id"],
        "user_id": upload_result["login_example"],
        "upload_date": upload_result["upload_date"],
        "position": upload_result["profile_obj"]["position"]["name"],
        "industries": upload_result["profile_obj"]["industries"],
        "time_experience_months": upload_result["profile_obj"]["position"]["time_experience_months"],
        "hard_skills": upload_result["profile_obj"]["hard_skills"],
        "soft_skills": upload_result["profile_obj"]["soft_skills"]
    }


@router.get("/resumes/{resume_id}", status_code=status.HTTP_200_OK)
async def get_resume_by_id(resume_id: str):
    try:
        resume_obj = get_resume_obj(resume_id)
    except ResumeNotFoundError(resume_id=resume_id) as e:
        raise HTTPException(status_code=404, detail=e.detail)
    return resume_obj


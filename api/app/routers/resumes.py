from fastapi import Depends, status, HTTPException, APIRouter, UploadFile
from api.app.services.resume_service import *
from api.app.dependencies import get_db, get_genai_client
from api.app.exceptions import ResumeParsingError, StructuredOutputParsingError, EmbeddingError, ResumeNotFoundError

router = APIRouter()


@router.post("/resumes/upload", status_code=status.HTTP_201_CREATED, response_model=UploadResponse)
async def send_resume(
    file: UploadFile,
    db: DatabaseManager = Depends(get_db),
    client: genai.Client = Depends(get_genai_client),
):
    try:
        return await process_resume_upload(client, db, file)
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


@router.get("/resumes/{resume_id}", status_code=status.HTTP_200_OK, response_model=ResumeResponse)
async def get_resume_by_id(
    resume_id: str,
    db: DatabaseManager = Depends(get_db)
):
    try:
        return get_resume_obj(db, resume_id)
    except ResumeNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)


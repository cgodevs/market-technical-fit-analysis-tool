from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from services.resume_service import process_resume_upload, get_upload_status, get_resume_obj
from database.manager import DatabaseManager
from dependencies import get_db


router = APIRouter(prefix="/resumes", tags=["resumes"])

@router.post("/upload", status_code=202)
async def upload_resume(file: UploadFile = File(...)):
    return await process_resume_upload(file)

@router.get("/{upload_id}/status")
def resume_status(upload_id: str):
    status = get_upload_status(upload_id)
    
    if status["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Resume processing failed: {status.get('error', 'unknown error')}"
        )
    
    return status

@router.get("/{resume_id}")
def get_resume(resume_id: str, db: DatabaseManager = Depends(get_db)):
    return get_resume_obj(db, resume_id)

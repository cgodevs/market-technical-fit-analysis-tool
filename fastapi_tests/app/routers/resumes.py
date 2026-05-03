from .. import analysis_utils as utils
from fastapi import UploadFile, status, HTTPException, APIRouter

router = APIRouter()


@router.post("/resumes/upload", status_code=status.HTTP_201_CREATED)
async def send_resume(file: UploadFile):
    return file.filename


@router.get("/resumes/{resume_id}", status_code=status.HTTP_200_OK)
async def candidate_resume(resume_id: str):
    db = utils.DatabaseManager()
    try:
        resume_df = db.get_resume(resume_id)
        resume_obj = resume_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Resume with id {resume_id} not found") from e
    finally:
        db.close_all()
    return resume_obj
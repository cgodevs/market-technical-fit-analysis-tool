from .. import utils
from fastapi import UploadFile, status, HTTPException, APIRouter

router = APIRouter()


@router.post("/resumes/upload", status_code=status.HTTP_201_CREATED)
async def send_resume(file: UploadFile):
    return file.filename


@router.get("/resumes/{resume_id}", status_code=status.HTTP_200_OK)
async def candidate_resume(resume_id: str):
    db = utils.DatabaseManager()
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
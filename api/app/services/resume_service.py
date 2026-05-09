import uuid
import pymupdf
import pymupdf4llm
from fastapi import UploadFile
from exceptions import ResumeParsingError, ResumeNotFoundError
from database.manager import DatabaseManager
from models.responses import ResumeResponse, AcceptedResponse
from tasks import process_resume_task, get_status

async def parse_resume(file: UploadFile):
    try:
        file_bytes = await file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        cv_md_text = pymupdf4llm.to_markdown(doc)
    except Exception as e:
        raise ResumeParsingError(detail=str(e))
    return cv_md_text

def get_resume_obj(db: DatabaseManager, resume_id: str) -> ResumeResponse:
    resume_df = db.get_resume(resume_id).drop(columns=["position_embedding"])
    if resume_df.empty:
        raise ResumeNotFoundError(resume_id=resume_id)
    row = resume_df.to_dict(orient="records")[0]
    return ResumeResponse(**row)

async def process_resume_upload(file: UploadFile) -> AcceptedResponse:
    """
    1. Parse the PDF to markdown (fast, ~seconds).
    2. Generate an upload_id.
    3. Enqueue the heavy work (LLM + embeddings + DB) to Celery.
    4. Return 202 Accepted with the upload_id for polling.
    """
    cv_md_text = await parse_resume(file)
    upload_id = str(uuid.uuid4())

    process_resume_task.delay(upload_id, cv_md_text)

    return AcceptedResponse(upload_id=upload_id)

def get_upload_status(upload_id: str) -> dict:
    status = get_status(upload_id)
    if status is None:
        raise ResumeNotFoundError(resume_id=upload_id)
    return status

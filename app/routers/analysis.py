from fastapi import Depends, status, HTTPException, APIRouter
from exceptions import EmbeddingError
from services.analysis_service import *
from dependencies import get_db
from models.responses import SkillsCoverageResponse

router = APIRouter()

@router.get("/analysis/compliant_coverage", status_code=status.HTTP_200_OK, response_model=SkillsCoverageResponse)
async def get_compliant_coverage(
    resume_id: str,
    db: DatabaseManager = Depends(get_db)
):
    try:
        return get_compliant_skills_coverage(db, resume_id)
    except ResumeNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except DatabaseQueryError as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {e.detail}")
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e.detail}")
    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e.detail}")
    except ResumeProcessingError as e:
        raise HTTPException(status_code=500, detail=f"Resume processing error: {e.detail}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate the compliance coverage for hard and soft skills. Error: {e}")

@router.get("/analysis/noncompliant_coverage", status_code=status.HTTP_200_OK, response_model=SkillsCoverageResponse)
async def get_noncompliant_coverage(
    resume_id: str,
    db: DatabaseManager = Depends(get_db)
):
    try:
        return get_noncompliant_skills_coverage(db, resume_id)
    except ResumeNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.detail)
    except DatabaseQueryError as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {e.detail}")
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e.detail}")
    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e.detail}")
    except ResumeProcessingError as e:
        raise HTTPException(status_code=500, detail=f"Resume processing error: {e.detail}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate the noncompliance coverage for hard and soft skills. Error: {e}")

@router.get("/analysis/breakdown", status_code=status.HTTP_200_OK)
async def get_compliance_breakdown(
    resume_id: str,
    db: DatabaseManager = Depends(get_db)
):
    pass

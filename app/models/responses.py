from pydantic import BaseModel
from datetime import datetime

class UploadResponse(BaseModel):
    resume_id: str
    upload_id: str
    user_id: str
    upload_date: str
    position: str
    industries: list[str]
    time_experience_months: int
    hard_skills: list
    soft_skills: list

class ResumeResponse(BaseModel):
    id: str
    user_id: str
    upload_id: str
    upload_date: datetime
    description: str
    industries: list[str]
    position: str
    time_experience_months: int


class SkillClusterSchema(BaseModel):
    skill_variants: list[str]
    total_matches: int
    unique_jobs: int
    job_coverage_pct: float

class SkillsCoverageResponse(BaseModel):
    soft_skills: list[SkillClusterSchema]
    hard_skills: list[SkillClusterSchema]

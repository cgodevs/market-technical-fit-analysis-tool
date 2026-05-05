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


class AnalysisDisplayResponse(BaseModel):
    job_id: str
    job_index: int
    required_skills: int
    matched_count: int
    minimum_compliance_pct: float
    matched_skills: list[str]
    insufficient_count: int
    ideal_compliance_pct: float
    insufficient_proficiency: list[str]
    nonmatched_count: int
    nonmatched_skills: list[str]

class PaginatedAnalysisDisplayResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    results: list[AnalysisDisplayResponse]

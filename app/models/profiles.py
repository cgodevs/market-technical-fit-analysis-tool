
from pydantic import BaseModel
from typing import List, Optional
from pydantic import BaseModel, Field
from ..config import SENIORITY_LEVELS


class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience_months: Optional[float] = Field(description="Experience in months, if explicitly stated, otherwise set to 0.")
    weight: Optional[float] = Field(description="Relevance experience score 0–1 for this position.")

class SoftSkill(BaseModel):
    description: str = Field(description="Canonical name of the soft skill in English (e.g. 'Stakeholder Communication').")
    weight: Optional[float] = Field(description="Relevance experience score 0–1 for this position.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or most experienced position")
    time_experience_months: Optional[float] = Field(description="Time experience in months identified for the held position. Set to 0 if not able to identify.")

class ProfessionalProfile(BaseModel):
    industries: List[str] = Field(description="Maximum of 2 matching LinkedIn industries list for the current goal job title, not the experience. Must be chosen from the list provided.")
    seniority: str = Field(description=f"Seniority level identifified for main goal position. Must be one of: {', '.join(SENIORITY_LEVELS)}.")
    position: Position
    hard_skills: List[HardSkill]
    soft_skills: List[SoftSkill]

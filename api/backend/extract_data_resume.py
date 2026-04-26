
import os
from mistune import markdown
import pymupdf4llm
from typing import List, Optional
from pydantic import BaseModel, Field

from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model # Added for generic model initialization

LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.2
api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=api_key)

class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience: Optional[float] = Field(description="Experience in months, if explicitly stated.")
    weight: Optional[float] = Field(description="Relevance score 0–1 for this position.")

class SoftSkill(BaseModel):
    description: str = Field(description="Canonical name of the soft skill in English (e.g. 'Stakeholder Communication').")
    weight: Optional[float] = Field(description="Relevance score 0–1 for this position.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or experienced position")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the held position.")

class ProfessionalProfile(BaseModel):
    industry: List[str] = Field(description="Maximum of 3 matching industries for the job seeking or experienced position.")
    positions: List[Position] = Field(description="Maximum of 3 canonical titles for the main matching positions identified for profile description and its variations. Example: Senior Accountant, Accountant III, Accountant Specialist, Experienced Accountant. The first one is the most relevant one.")
    hard_skills: List[HardSkill]
    soft_skills: List[SoftSkill]


def extract_professional_structured_data(text: str) -> dict:
    llm = init_chat_model(
        model=LLM_MODEL_NAME,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your role is to extract relevant data for building a structures object of metadata for a professional resume"),
        ("human", "{input}")
    ])

    structured_llm = llm.with_structured_output(schema=ProfessionalProfile)
    chain = prompt | structured_llm
    response = chain.invoke({"input": text})
    return response.model_dump()

path_my_cv = "./api/data/cv_example.pdf"
cv_md_text = pymupdf4llm.to_markdown(path_my_cv)

cv_obj = extract_professional_structured_data(cv_md_text)
print(cv_obj)


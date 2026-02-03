from database_mock import JOBS_DB

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import List, Optional
from pydantic import BaseModel, Field
import streamlit as st

class HardSkill(BaseModel):
    description: str = Field(description="Description of the hard skill.")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the hard skill application.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or experienced position")
    field: str = Field(description="Field of the position")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the held position.")

class ProfessionalProfile(BaseModel):
    positions: List[Position] = Field(description="Maximum of 3 names for matching positions identified in the profile description")
    main_position: List[Position] = Field(description="Main position goal identified for profile description")
    main_position_name_variations: list = Field(description="Main position name variations. Example: Senior Accountant, Accountant III, Accountant Specialist, Experienced Accountant")
    hard_skills: List[HardSkill]
    soft_skills: List[str] = Field(description="Description of the soft skill")


def extract_professional_structured_data(text: str, api_key: str) -> dict:
    """Uses Google GenAI to extract either the CV profile or Job description required profile."""
    try:
        # Initialize Google GenAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # Using a fast model for the prototype
            temperature=0,
            google_api_key=api_key,
            max_retries=2
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Your role is to extract relevant data for building a professional role description object out of either a curriculum vitae text or a description for a job position out of input text"),
            ("human", "{input}")
        ])

        structured_llm = llm.with_structured_output(schema=ProfessionalProfile)
        chain = prompt | structured_llm
        response = chain.invoke({"input": text})
        return response.model_dump()
        
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")
        return None
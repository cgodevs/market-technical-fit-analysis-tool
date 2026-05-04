import pymupdf
import pymupdf4llm
from langchain.chat_models import init_chat_model
from fastapi import UploadFile
from langchain_core.prompts import ChatPromptTemplate
from google import genai

from app.database.manager import DatabaseManager
from backend.extract_data_resume import RESUMES_TABLE
from ..models.profiles import ProfessionalProfile
from ..config import api_key, LLM_MODEL_NAME, LLM_PROVIDER, LLM_TEMPERATURE, CANDIDATE_HARD_SKILLS_TABLE, CANDIDATE_SOFT_SKILLS_TABLE
from ..utils.db_utils import get_static_list_of_industries, save_df_to_database
from ..utils.embeddings import recreate_df_with_embeddings
import pandas as pd


async def parse_resume(file: UploadFile):
    try:
        file_bytes = await file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        cv_md_text = pymupdf4llm.to_markdown(doc)
    except Exception:
        cv_md_text = ""
    return cv_md_text

def extract_professional_structured_data(text: str) -> dict:
    llm = init_chat_model(
        model=LLM_MODEL_NAME,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=api_key
    )
    system_prompt = f"""
        Your role is to extract data out of a resume text provided to build it a metadata object. 
        Use all sets of experiences identified to build a complete object.
        Work industries list to choose from for the main goal position: {'|'.join(get_static_list_of_industries())}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    structured_llm = llm.with_structured_output(schema=ProfessionalProfile)
    chain = prompt | structured_llm
    response = chain.invoke({"input": text})
    return response.model_dump()

def get_resume_obj(resume_id: str) -> dict:
    db = DatabaseManager()
    try:
        resume_df = db.get_resume(resume_id).drop(columns=["position_embedding"])
        resume_obj = resume_df.to_dict(orient="records")[0] if not resume_df.empty else {}
    except Exception as e:
        resume_obj = {}
    finally:
        db.close_all()
    return resume_obj

def save_resume_to_database(cv_obj: dict):
    cv_df = pd.DataFrame([cv_obj])
    save_df_to_database(df=cv_df, table_name=RESUMES_TABLE)

def save_skills_to_database(profile_obj: dict, resume_upload_id: str):
    client = genai.Client(api_key=api_key)

    hard_skills_df = pd.DataFrame(profile_obj["hard_skills"])
    soft_skills_df = pd.DataFrame(profile_obj["soft_skills"])

    hard_skills_df = recreate_df_with_embeddings(client, hard_skills_df, "description")
    soft_skills_df = recreate_df_with_embeddings(client, soft_skills_df, "description")

    hard_skills_df["resume_id"] = resume_upload_id
    soft_skills_df["resume_id"] = resume_upload_id

    hard_skills_df = hard_skills_df[["resume_id", "description", "weight", "time_experience_months", "embedding"]]
    soft_skills_df = soft_skills_df[["resume_id", "description", "weight", "embedding"]]

    save_df_to_database(df=hard_skills_df, table_name=CANDIDATE_HARD_SKILLS_TABLE)
    save_df_to_database(df=soft_skills_df, table_name=CANDIDATE_SOFT_SKILLS_TABLE)
import uuid
import pymupdf
import pymupdf4llm
import pandas as pd
from langchain.chat_models import init_chat_model
from fastapi import UploadFile
from langchain_core.prompts import ChatPromptTemplate
from google import genai
from random import randint
from exceptions import ResumeParsingError, ResumeNotFoundError, StructuredOutputParsingError, ResumeProcessingError
from database.manager import DatabaseManager
from models.profiles import ProfessionalProfile
from models.responses import UploadResponse, ResumeResponse
from config import api_key, LLM_MODEL_NAME, LLM_PROVIDER, LLM_TEMPERATURE
from utils.db_utils import get_static_list_of_industries, save_resume_data
from utils.embeddings import df_with_embedding_column, embed_texts
from datetime import datetime


async def parse_resume(file: UploadFile):
    try:
        file_bytes = await file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        cv_md_text = pymupdf4llm.to_markdown(doc)
    except Exception as e:
        raise ResumeParsingError(detail=str(e))
    return cv_md_text

def extract_professional_structured_data(text: str) -> dict:
    try:
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
        data= response.model_dump()
    except Exception as e:
        raise StructuredOutputParsingError(detail=str(e))
    return data

def get_resume_obj(db: DatabaseManager, resume_id: str) -> ResumeResponse:
    try:
        resume_df = db.get_resume(resume_id).drop(columns=["position_embedding"])
        if resume_df.empty:
            raise ResumeNotFoundError(resume_id=resume_id)
        row = resume_df.to_dict(orient="records")[0]
        return ResumeResponse(**row)
    except ResumeNotFoundError:
        raise
    except Exception:
        raise ResumeNotFoundError(resume_id=resume_id)

async def process_resume_upload(client: genai.Client, db: DatabaseManager, file: UploadFile):
    cv_md_text = await parse_resume(file)
    profile_obj = extract_professional_structured_data(cv_md_text)
    position_embedding = embed_texts(client, [profile_obj["position"]["name"]])[0]

    login_example = f"person{randint(0, 9999)}@mail.com"
    resume_upload_id = str(uuid.uuid4())
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv_obj = {
        "id": resume_upload_id,
        "user_id": login_example,
        "upload_id": resume_upload_id,
        "upload_date": upload_date,
        "description": cv_md_text,
        "industries": profile_obj["industries"],
        "position": profile_obj["position"]["name"],
        "time_experience_months": profile_obj["position"]["time_experience_months"],
        "position_embedding": position_embedding
    }

    cv_df = pd.DataFrame([cv_obj])
    hard_skills_df = pd.DataFrame(profile_obj["hard_skills"])
    soft_skills_df = pd.DataFrame(profile_obj["soft_skills"])

    hard_skills_df = df_with_embedding_column(client, hard_skills_df, "description")
    soft_skills_df = df_with_embedding_column(client, soft_skills_df, "description")

    hard_skills_df["resume_id"] = resume_upload_id
    soft_skills_df["resume_id"] = resume_upload_id

    hard_skills_df = hard_skills_df[["resume_id", "description", "weight", "time_experience_months", "embedding"]]
    soft_skills_df = soft_skills_df[["resume_id", "description", "weight", "embedding"]]

    try:
        with db.get_conn() as conn:
            save_resume_data(
                conn=conn,
                cv_df=cv_df,
                hard_skills_df=hard_skills_df,
                soft_skills_df=soft_skills_df
            )
    except Exception as e:
        print(f"DB save error: {type(e).__name__}: {e}")
        raise ResumeProcessingError(detail="Failed to save resume data to database")

    
    return UploadResponse(
        upload_id=resume_upload_id,
        resume_id=resume_upload_id,
        user_id=login_example,
        upload_date=upload_date,
        position=profile_obj["position"]["name"],
        industries=profile_obj["industries"],
        time_experience_months=profile_obj["position"]["time_experience_months"],
        hard_skills=profile_obj["hard_skills"],
        soft_skills=profile_obj["soft_skills"]
    )

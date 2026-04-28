import os
import re
import time
import pymupdf4llm
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from typing import List, Optional
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

PATH_EXAMPLE_CV = "./api/data/cv_example.pdf"

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
EMBED_BATCH_SIZE = 100
EMBED_CONCURRENCY = 5
EMBED_MAX_RETRIES = 3

DB_NAME = "market_fit"
DB_HOST = "localhost"
db_user = os.getenv("DB_USER")
db_pw = os.getenv("DB_PASSWORD")
api_key = os.getenv('GEMINI_API_KEY')
RESUMES_TABLE = "resumes"
CANDIDATE_HARD_SKILLS_TABLE = "candidate_hard_skills"
CANDIDATE_SOFT_SKILLS_TABLE = "candidate_soft_skills"
SENIORITY_LEVELS = (
    "Intern", "Junior", "Mid", "Senior", "Associate", "Specialist",
    "Manager", "Director", "Head", "President/Vice President",
    "C-Level", "Partner", "Owner", "Founder",
)


class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience: Optional[float] = Field(description="Experience in months, if explicitly stated.")
    weight: Optional[float] = Field(description="Relevance experience score 0–1 for this position.")

class SoftSkill(BaseModel):
    description: str = Field(description="Canonical name of the soft skill in English (e.g. 'Stakeholder Communication').")
    weight: Optional[float] = Field(description="Relevance experience score 0–1 for this position.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or most experienced position")
    time_experience: Optional[float] = Field(description="Time experience in months identified for the held position.")

class ProfessionalProfile(BaseModel):
    industries: List[str] = Field(description="Maximum of 3 matching LinkedIn industries list for the current goal job. Must be chosen from the list provided.")
    seniority: str = Field(description=f"Seniority level identifified for main goal position. Must be one of: {', '.join(SENIORITY_LEVELS)}.")
    position: Position
    hard_skills: List[HardSkill]
    soft_skills: List[SoftSkill]


def get_list_of_industries_from_local_file() -> List[str]:
    with open("./api/data/enum_industry.txt", "r") as f:
        industries = [line.strip() for line in f if line.strip()]
        industries = [industry.upper() for industry in industries] 
    return industries

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
        Work industries list to choose from for the main goal position: {'|'.join(get_list_of_industries_from_local_file())}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    structured_llm = llm.with_structured_output(schema=ProfessionalProfile)
    chain = prompt | structured_llm
    response = chain.invoke({"input": text})
    return response.model_dump()

def _embed_batch_with_retry(
    client: genai.Client,
    texts: list[str],
    max_retries: int = EMBED_MAX_RETRIES,
) -> list[list[float]]:
    """Embed a batch of strings, retrying on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
            return [e.values for e in response.embeddings]
        except genai.errors.ClientError as exc:
            is_rate_limit = exc.code == 429 or "RESOURCE_EXHAUSTED" in str(exc.status)
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
            wait = float(match.group(1)) if match else 2 ** attempt * 10
            print(f"Rate limited — waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

def embed_texts(
    client: genai.Client,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
    concurrency: int = EMBED_CONCURRENCY,
) -> list[list[float]]:
    """Embed an arbitrary list of texts in parallel batches."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_embed_batch_with_retry, client, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [emb for batch in results for emb in batch]

def recreate_df_with_embeddings(client: genai.Client, df: pd.DataFrame, column_to_embed: str, batch_size: int = 100, concurrency: int = 5) -> pd.DataFrame:
    result_df = df.copy()
    skills = result_df[column_to_embed].tolist()
    batches = [skills[i:i + batch_size] for i in range(0, len(skills), batch_size)]

    with ThreadPool(concurrency) as pool:
        batch_results = pool.map(lambda batch: _embed_batch_with_retry(client, batch), batches)

    embeddings = [emb for batch in batch_results for emb in batch]
    result_df["embedding"] = embeddings
    return result_df

def save_df_to_database(df: pd.DataFrame, table_name: str) -> int:
    cols = ", ".join(df.columns)
    placeholders = ", ".join(
        "%s::vector" if "embedding" in col else "%s"
        for col in df.columns
    )
    template = f"({placeholders})"

    with psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=db_user, password=db_pw
    ) as conn:
        conn.autocommit = False
        cur = conn.cursor()
        execute_values(
            cur,
            f"INSERT INTO {table_name} ({cols}) VALUES %s",
            [tuple(row) for _, row in df.iterrows()],
            template=template,
            page_size=500
        )
        ids = [row[0] for row in cur.fetchall()]
        conn.commit()
        return ids[0]



# Update vars for testing
EMAIL_EXAMPLE = "caroline.development@gmail.com"
RESUME_UPLOAD_ID = '1'

client = genai.Client(api_key=api_key)

cv_md_text = pymupdf4llm.to_markdown(PATH_EXAMPLE_CV)
profile_obj = extract_professional_structured_data(cv_md_text)
upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

cv_obj = {
    "user_id": EMAIL_EXAMPLE,
    "upload_id": RESUME_UPLOAD_ID,
    "upload_date": upload_date,
    "description": cv_md_text,
    "industries": profile_obj["industries"],
    "position": profile_obj["position"]["name"],
    "time_experience_months": profile_obj["position"]["time_experience"],
    "position_embedding": embed_texts(client, [profile_obj["position"]["name"]])[0]
}

hard_skills_obj = {
    "resume_id": RESUME_UPLOAD_ID,
    "weight": [skill["weight"] for skill in profile_obj["hard_skills"]],
    "description": [skill["description"] for skill in profile_obj["hard_skills"]],
    "time_experience_months": [skill["time_experience"] for skill in profile_obj["hard_skills"]]
}

soft_skills_obj = {
    "resume_id": RESUME_UPLOAD_ID,
    "weight": [skill["weight"] for skill in profile_obj["soft_skills"]],
    "description": [skill["description"] for skill in profile_obj["soft_skills"]]
}


cv_df = pd.DataFrame([cv_obj])
hard_skills_df = pd.DataFrame(profile_obj["hard_skills"])
soft_skills_df = pd.DataFrame(profile_obj["soft_skills"])

hard_skills_df = recreate_df_with_embeddings(client, hard_skills_df, "description")
soft_skills_df = recreate_df_with_embeddings(client, soft_skills_df, "description")

# resume_id = save_df_to_database(df=cv_df, table_name=RESUMES_TABLE)
# hard_skills_obj["id"] = resume_id
# soft_skills_obj["id"] = resume_id

# save_df_to_database(df=hard_skills_df, table_name=CANDIDATE_HARD_SKILLS_TABLE)
# save_df_to_database(df=soft_skills_df, table_name=CANDIDATE_SOFT_SKILLS_TABLE)

# TODO: Recreate database tables for soft skills extracted from resumes and hard skills too (same columns as the equivalent skills extracted from job descriptions)
# TODO: Filter database by industries
# TODO: For each job, use positions embeddings for the title to calculate similarity with positions_embeddings to find the best matching positions to bring up for analysis and its time experience.
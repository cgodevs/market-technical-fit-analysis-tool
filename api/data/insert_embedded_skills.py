'''
-- Create tables for hard and soft skills with embeddings

CREATE TABLE soft_skills (
    id          SERIAL PRIMARY KEY,
    job_id      VARCHAR(255) NOT NULL REFERENCES job_postings(id) ON DELETE CASCADE,
    skill_description TEXT NOT NULL,
    weight      FLOAT,
    embedding   VECTOR(3072)  -- gemini-embedding-001 output dimension
);

CREATE TABLE hard_skills (
    id              SERIAL PRIMARY KEY,
    job_id          VARCHAR(255) NOT NULL REFERENCES job_postings(id) ON DELETE CASCADE,
    skill_description TEXT NOT NULL,
    time_experience FLOAT,
    weight          FLOAT,
    embedding       VECTOR(3072)
);
'''

from pgvector.psycopg2 import register_vector
import psycopg2
import pandas as pd
import psycopg2
from os import getenv
from typing import List, Optional
from google import genai
from pydantic import BaseModel, Field
from multiprocessing.pool import ThreadPool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model 
import time
import re
from google.genai import errors
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_REASONING_LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
HOST = "localhost"
DATABASE = "market_fit"
JOB_POSTINGS_TABLE_NAME = "job_postings"
api_key = getenv('GEMINI_API_KEY')
database_user = getenv("DB_USER")
database_password = getenv("DB_PASSWORD")


class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the hard skill application.")
    weight: Optional[float] = Field(description="Optional weighted relevance score for the hard skill application in general for the position, ranging from 0 to 1.")

class HardSkillList(BaseModel):
    hard_skills: List[HardSkill] = Field(description="List of hard skills extracted from a job description.")

class SoftSkill(BaseModel):
    description: str = Field(description="Concise canonical name of the soft skill in English (e.g. 'Stakeholder Communication', 'Proactive Communication').")
    weight: Optional[float] = Field(description="Weighted relevance score for this soft skill for the position, ranging from 0 to 1.")

class SoftSkillList(BaseModel):
    soft_skills: List[SoftSkill] = Field(description="List of soft skills extracted from a job description.")

class SkillsList(BaseModel):
    hard_skills: List[HardSkill] = Field(description="List of hard skills extracted from a job description.")
    soft_skills: List[SoftSkill] = Field(description="List of soft skills extracted from a job description.")


def _build_llm(model_name=DEFAULT_REASONING_LLM_MODEL) -> genai.Client:
    return init_chat_model(
        model=model_name,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=api_key
    )

def embed_skill(chat_model: genai.Client, skill: str) -> List[float]:
    try:
        response = chat_model.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[skill]
        )
    except Exception as e:
        print(f"Error embedding skill '{skill}': {e}")
        return []
    return response.embeddings[0].values

def extract_skills_list(chat_model: genai.Client, text: str) -> SkillsList:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are an expert technical recruiter and organizational psychologist specializing in behavioral competency frameworks.
            Your task is to extract ALL hard and soft skills from a job description and assign each a relevance weight.

            ## Hard Skills
            Extract specific tools, technologies, methodologies, and domain knowledge.

            ### Rules
            1. Extract BOTH the broad category AND each specific tool/technology mentioned within it.
               - Example: "ETL (SSIS or Azure)" → extract `ETL`, `SSIS`, and `Azure` as separate entries.
            2. Use concise canonical names in English (e.g. `SQL`, `Python`, `ETL`, `Azure Data Factory`).
               - If a skill implies a category, extract both (e.g. "Programming language (Python preferred)" → `Programming` + `Python`).
               - For architecture/design skills use compound form: "data pipeline and analytics architectures" → `Data Pipeline Architecture` + `Data Architecture`.
               - Not too general (`Microsoft Cloud Services` not `Microsoft`), not too specific (`Python` not `Python 3.8`).
            3. Always extract language requirements as hard skills (`Fluent English`, `English Proficiency`).
            4. `time_experience`: only populate if the job explicitly mentions a duration (e.g. "3+ years of Python") or interpret it as the general time of experience for the role, if mentioned. Otherwise null.

            ## Soft Skills
            Extract interpersonal, behavioral, or cognitive traits NOT tied to a specific tool, technology, or domain.
            Examples: `Stakeholder Communication`, `Cross-functional Collaboration`, `Proactive Communication`.

            ### What does NOT count as a soft skill
            - Tool or technology proficiency (e.g. SQL, Python) → hard skill
            - Language proficiency (e.g. Fluent English) → hard skill
            - Domain knowledge (e.g. Data Modeling, ETL) → hard skill

            ### Rules
            1. Use short noun phrases in English (e.g. `Stakeholder Communication`, `Proactive Risk Escalation`).
            2. If a sentence implies multiple distinct soft skills, extract each separately.

            ## Weight Assignment (applies to both hard and soft skills)
            | Signal in job description                        | Weight range |
            |--------------------------------------------------|--------------|
            | Required / mandatory / strongly emphasized       | 0.8 – 1.0    |
            | Preferred / clearly mentioned                    | 0.5 – 0.7    |
            | Desirable / implied / nice-to-have               | 0.2 – 0.4    |

            - Parent categories of required tools inherit a high weight.
            - Specific tools listed as "preferred" get slightly lower weight than their parent category.
            - All descriptions must be in English, even if the job description is in another language.
        """),
        ("human", "{input}")
    ])

    structured_llm = chat_model.with_structured_output(schema=SkillsList)
    chain = prompt | structured_llm
    return chain.invoke({"input": text})

def process_row_skills(chat_model, row: dict) -> tuple:
    job_id = row['id']
    text = f"Job title: {row['title']}\nJob description: {row['description']}"
    try:
        skills = extract_skills_list(chat_model, text)
    except Exception as e:
        print(f"Error processing job_id {job_id}: {e}")
        skills = SkillsList(hard_skills=[], soft_skills=[])
    return job_id, skills

def build_skills_dfs(chat_model, df: pd.DataFrame, concurrency: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = df.to_dict(orient='records')

    with ThreadPool(concurrency) as pool:
        results = pool.map(lambda row: process_row_skills(chat_model, row), rows)

    hard_records, soft_records = [], []
    for job_id, skills in results:
        for skill in skills.hard_skills:
            hard_records.append({
                "job_id": job_id,
                "skill_description": skill.description,
                "time_experience": skill.time_experience,
                "weight": skill.weight
            })
        for skill in skills.soft_skills:
            soft_records.append({
                "job_id": job_id,
                "skill_description": skill.description,
                "weight": skill.weight
            })

    return pd.DataFrame(hard_records), pd.DataFrame(soft_records)

def embed_batch_with_retry(client: genai.Client, skills: List[str], max_retries: int = 3) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=skills
            )
            return [e.values for e in response.embeddings]
        except errors.ClientError as e:
            is_rate_limit = (e.code == 429 or "RESOURCE_EXHAUSTED" in str(e.status))
            if not is_rate_limit or attempt == max_retries - 1:
                raise

            match = re.search(r'retry in (\d+(?:\.\d+)?)s', str(e))
            wait = float(match.group(1)) if match else (2 ** attempt * 10)
            print(f"Rate limited. Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)

    raise RuntimeError("Max retries exceeded")

def build_embeddings(client: genai.Client, df: pd.DataFrame, batch_size: int = 100, concurrency: int = 5) -> pd.DataFrame:
    result_df = df.copy()
    skills = result_df["skill_description"].tolist()

    batches = [skills[i:i + batch_size] for i in range(0, len(skills), batch_size)]

    with ThreadPool(concurrency) as pool:
        batch_results = pool.map(lambda batch: embed_batch_with_retry(client, batch), batches)

    embeddings = [emb for batch in batch_results for emb in batch]
    result_df["embedding"] = embeddings
    return result_df

def save_hard_skills(hard_skills_df: pd.DataFrame):
    with psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=database_user,
        password=database_password
    ) as conn:
        if conn:
            print("Connection to the database was successful!")
            conn.autocommit = True
            register_vector(conn)

        cur = conn.cursor()
        for _, row in hard_skills_df.iterrows():
            insert_query = """
                INSERT INTO hard_skills (job_id, skill_description, time_experience, weight, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """
            values = (
                row["job_id"],
                row["skill_description"],
                row.get("time_experience"),
                row.get("weight"),
                row["embedding"]
            )
            cur.execute(insert_query, values)

def save_soft_skills(soft_skills_df: pd.DataFrame):
    with psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=database_user,
        password=database_password
    ) as conn:
        if conn:
            print("Connection to the database was successful!")
            conn.autocommit = True
            register_vector(conn)

        cur = conn.cursor()
        for _, row in soft_skills_df.iterrows():
            insert_query = """
                INSERT INTO soft_skills (job_id, skill_description, weight, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """
            values = (
                row["job_id"],
                row["skill_description"],
                row.get("weight"),
                row["embedding"]
            )
            cur.execute(insert_query, values)


# ── Run ───────────────────────────────────────────────────────────────────────

with psycopg2.connect(
    host=HOST,
    database=DATABASE,
    user=database_user,
    password=database_password
) as conn:
    if not conn:
        print("Connection to the database failed!")
        exit(1)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {JOB_POSTINGS_TABLE_NAME}")  
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    df = df[["id", "title", "description"]]

overall_start_time = time.time()
chat_model = _build_llm()
hard_skills_df, soft_skills_df = build_skills_dfs(chat_model, df)  
client = genai.Client(api_key=api_key)

with ThreadPoolExecutor(max_workers=2) as executor:
    hard_emb_future = executor.submit(build_embeddings, client, hard_skills_df)
    soft_emb_future = executor.submit(build_embeddings, client, soft_skills_df)
    hard_skills_df_with_embeddings = hard_emb_future.result()
    soft_skills_df_with_embeddings = soft_emb_future.result()

print(f"Total processing time: {(time.time() - overall_start_time)/60:.2f} minutes")
print(len(df), "job postings\n")
print(len(hard_skills_df), "hard skills extracted")
print(len(soft_skills_df), "soft skills extracted")

with ThreadPoolExecutor(max_workers=2) as executor:
    hard_future = executor.submit(save_hard_skills, hard_skills_df_with_embeddings)
    soft_future = executor.submit(save_soft_skills, soft_skills_df_with_embeddings)
    hard_future.result()
    soft_future.result()

import pandas as pd
import psycopg2
from langchain.chat_models import init_chat_model 
from os import getenv
from typing import Optional
from pydantic import BaseModel, Field
from multiprocessing.pool import ThreadPool
from langchain_core.prompts import ChatPromptTemplate
from google import genai


DEFAULT_REASONING_LLM_MODEL = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
HOST = "localhost"
DATABASE = "market_fit"
JOB_POSTINGS_TABLE_NAME = "job_postings"
api_key = getenv('GEMINI_API_KEY')
database_user = getenv("DB_USER")
database_password = getenv("DB_PASSWORD")


class SeniorityExtraction(BaseModel):
    min_seniority: str = Field(description="Minimum seniority level required for the position. Must be one of: Intern, Junior, Mid, Senior, Associate, Specialist, Manager, Director, Head, President/Vice President, C-Level, Partner, Owner, Founder.")
    time_experience_months: Optional[float] = Field(description="Minimum time of experience in months explicitly or implicitly required for the position. Null if not mentioned.")

def _build_llm(model_name=DEFAULT_REASONING_LLM_MODEL) -> genai.Client:
    return init_chat_model(
        model=model_name,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=api_key
    )
    
def extract_seniority(chat_model, text: str) -> SeniorityExtraction:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are an expert technical recruiter specializing in seniority assessment.
            Your task is to extract the MINIMUM seniority level and experience time required for a job position.

            ## Seniority Levels (ordered from lowest to highest)
            Intern, Junior, Mid, Senior, Associate, Specialist, Manager, Director, Head, President/Vice President, C-Level, Partner, Owner, Founder

            ## Rules
            0. Sometimes the Seniority will be hardcoded in the input given. In that case, just return it without overthinking.
            1. Pick the MINIMUM seniority level that would be accepted for the role — not the ideal candidate.
            2. If the job says "Senior or above", the minimum is Senior.
            3. If no seniority is explicitly mentioned, infer it from context:
               - Internship/trainee programs → Intern
               - "Entry level" or < 2 years experience → Junior
               - 2–4 years experience → Mid
               - 5+ years experience → Senior
               - People management responsibilities → Manager or above
            4. `time_experience_months`: convert years to months (e.g. "3 years" → 36.0).
               Only populate if a duration is explicitly stated or strongly implied. Otherwise null.
            5. Output must use exactly one of the allowed seniority strings.
        """),
        ("human", "{input}")
    ])

    structured_llm = chat_model.with_structured_output(schema=SeniorityExtraction)
    chain = prompt | structured_llm
    return chain.invoke({"input": text})

def process_row_seniority(chat_model, row: dict) -> tuple:
    job_id = row["id"]
    text = f"Job title: {row['title']}\nJob description: {row['description']}\nHardcoded seniority: {row['f_ai_min_seniority']}"
    try:
        result = extract_seniority(chat_model, text)
    except Exception as e:
        print(f"Error processing job_id {job_id}: {e}")
        result = SeniorityExtraction(min_seniority="Mid", time_experience_months=None)
    return job_id, result

def build_seniority_df(chat_model, df: pd.DataFrame, concurrency: int = 10) -> pd.DataFrame:
    rows = df.to_dict(orient="records")

    with ThreadPool(concurrency) as pool:
        results = pool.map(lambda row: process_row_seniority(chat_model, row), rows)

    records = []
    for job_id, result in results:
        records.append({
            "job_id": job_id,
            "f_ai_min_seniority": result.min_seniority,
            "ai_experience_time_months": result.time_experience_months
        })

    return pd.DataFrame(records)

chat_model = _build_llm()

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
    cur.execute(f"""
            SELECT 
                id, 
                title, 
                description,
                f_ai_min_seniority
            FROM {JOB_POSTINGS_TABLE_NAME}
        """)
    rows = cur.fetchall()
    jobs_df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    jobs_df["ai_experience_time_months"] = None

    df = build_seniority_df(chat_model=chat_model, df=jobs_df)
    df["ai_experience_time_months"].fillna(0, inplace=True)

    for _, row in df.iterrows():
        cur.execute(f"""
            UPDATE {JOB_POSTINGS_TABLE_NAME}
            SET f_ai_min_seniority = %s, 
                ai_experience_time_months = %s            
            WHERE id = %s
        """, (row["min_seniority"], row["time_experience_months"], row["job_id"]))



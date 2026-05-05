"""
Skills extraction pipeline: job_postings → hard_skills + soft_skills (with embeddings).

Stages:
  1. Load      – fetch job postings from PostgreSQL
  2. Extract   – LLM extracts hard/soft skills per job (parallel)
  3. Embed     – embed skill descriptions in parallel batches
  4. Save      – bulk-insert both skill tables (parallel)
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5

DB_HOST = "localhost"
DB_NAME = "market_fit"
DB_USER = getenv("DB_USER")
DB_PASSWORD = getenv("DB_PASSWORD")
GEMINI_API_KEY = getenv("GEMINI_API_KEY")

JOB_POSTINGS_TABLE = "job_postings"
HARD_SKILLS_TABLE = "hard_skills"
SOFT_SKILLS_TABLE = "soft_skills"

SKILLS_CONCURRENCY = 10
EMBED_CONCURRENCY = 5
EMBED_BATCH_SIZE = 100
EMBED_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience: Optional[float] = Field(description="Experience in months, if explicitly stated.")
    weight: Optional[float] = Field(description="Relevance score 0–1 for this position.")

class SoftSkill(BaseModel):
    description: str = Field(description="Canonical name of the soft skill in English (e.g. 'Stakeholder Communication').")
    weight: Optional[float] = Field(description="Relevance score 0–1 for this position.")

class SkillsList(BaseModel):
    hard_skills: list[HardSkill]
    soft_skills: list[SoftSkill]

# ---------------------------------------------------------------------------
# LLM skill extraction
# ---------------------------------------------------------------------------

_SKILLS_SYSTEM_PROMPT = """
    You are an expert technical recruiter and organizational psychologist specializing in behavioral competency frameworks.
    Extract ALL hard and soft skills from the job description and assign each a relevance weight.

    ## Hard Skills
    Specific tools, technologies, methodologies, and domain knowledge.

    Rules:
    1. Extract BOTH the broad category AND each specific tool mentioned.
    - e.g. "ETL (SSIS or Azure)" → `ETL`, `SSIS`, `Azure` as separate entries.
    2. Use concise canonical English names (e.g. `SQL`, `Python`, `Azure Data Factory`).
    - Not too general (`Microsoft Cloud Services`, not `Microsoft`).
    - Not too specific (`Python`, not `Python 3.8`).
    3. Always extract language requirements as hard skills (e.g. `Fluent English`).
    4. `time_experience`: populate only if a duration is explicitly mentioned; otherwise null.
    5. Specific tools listed as "preferred" get slightly lower weight than their parent category.

    ## Soft Skills
    Interpersonal, behavioral, or cognitive traits NOT tied to a specific tool or domain.
    Examples: `Stakeholder Communication`, `Cross-functional Collaboration`.
    NOT soft skills: tool proficiency, language proficiency, domain knowledge → those are hard skills.

    Rules:
    1. Use short English noun phrases (e.g. `Proactive Risk Escalation`).
    2. If a sentence implies multiple distinct soft skills, extract each separately.
    
    ## Weight assignment (both skill types)
    | Signal                                      | Range     |
    |---------------------------------------------|-----------|
    | Required / mandatory / strongly emphasized  | 0.8 – 1.0 |
    | Preferred / clearly mentioned               | 0.5 – 0.7 |
    | Desirable / implied / nice-to-have          | 0.2 – 0.4 |

    All descriptions must be in English, even if the source text is in another language.
"""


def _build_skills_chain(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SKILLS_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | chat_model.with_structured_output(schema=SkillsList)


def _extract_skills_for_row(chain, row: dict) -> tuple[str, SkillsList]:
    text = f"Job title: {row['title']}\nJob description: {row['description']}"
    try:
        skills = chain.invoke({"input": text})
    except Exception as exc:
        print(f"Skills extraction failed for job {row['id']}: {exc}")
        skills = SkillsList(hard_skills=[], soft_skills=[])
    return row["id"], skills


def extract_skills_bulk(
    chat_model,
    df: pd.DataFrame,
    concurrency: int = SKILLS_CONCURRENCY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract hard and soft skills for every job in df.
    Returns (hard_skills_df, soft_skills_df).
    """
    chain = _build_skills_chain(chat_model)
    rows = df.to_dict(orient="records")

    job_results: list[tuple[str, SkillsList]] = [None] * len(rows)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_extract_skills_for_row, chain, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            job_results[futures[future]] = future.result()

    hard_records, soft_records = [], []
    for job_id, skills in job_results:
        for skill in skills.hard_skills:
            hard_records.append({
                "job_id": job_id,
                "skill_description": skill.description,
                "time_experience": skill.time_experience,
                "weight": skill.weight,
            })
        for skill in skills.soft_skills:
            soft_records.append({
                "job_id": job_id,
                "skill_description": skill.description,
                "weight": skill.weight,
            })

    return pd.DataFrame(hard_records), pd.DataFrame(soft_records)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch_with_retry(
    client: genai.Client,
    texts: list[str],
    max_retries: int = EMBED_MAX_RETRIES,
) -> list[list[float]]:
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


def embed_column(
    client: genai.Client,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
    concurrency: int = EMBED_CONCURRENCY,
) -> list[list[float]]:
    """Embed an arbitrary list of strings in parallel batches, preserving order."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_embed_batch_with_retry, client, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [emb for batch in results for emb in batch]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def db_connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    register_vector(conn)
    return conn


def load_job_postings() -> pd.DataFrame:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, title, description FROM {JOB_POSTINGS_TABLE}")
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _bulk_insert(conn, table: str, columns: list[str], df: pd.DataFrame) -> None:
    records = [tuple(row[c] for c in columns) for _, row in df[columns].iterrows()]
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=500)
    conn.commit()
    print(f"Inserted {len(records)} rows into {table}.")


def save_hard_skills(df: pd.DataFrame) -> None:
    columns = ["job_id", "skill_description", "time_experience", "weight", "embedding"]
    with db_connect() as conn:
        _bulk_insert(conn, HARD_SKILLS_TABLE, columns, df)


def save_soft_skills(df: pd.DataFrame) -> None:
    columns = ["job_id", "skill_description", "weight", "embedding"]
    with db_connect() as conn:
        _bulk_insert(conn, SOFT_SKILLS_TABLE, columns, df)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()
    chat_model = init_chat_model(
        model=MODEL_NAME,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=GEMINI_API_KEY,
    )

    print("Loading job postings...")
    jobs_df = load_job_postings()
    print(f"  {len(jobs_df)} postings loaded.")

    print("Extracting skills with LLM...")
    hard_df, soft_df = extract_skills_bulk(chat_model, jobs_df)
    print(f"  {len(hard_df)} hard skills / {len(soft_df)} soft skills extracted.")

    print("Embedding skill descriptions (parallel)...")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    with ThreadPoolExecutor(max_workers=2) as pool:
        hard_future = pool.submit(embed_column, gemini_client, hard_df["skill_description"].tolist())
        soft_future = pool.submit(embed_column, gemini_client, soft_df["skill_description"].tolist())
        hard_df = hard_df.copy()
        soft_df = soft_df.copy()
        hard_df["embedding"] = hard_future.result()
        soft_df["embedding"] = soft_future.result()

    print("Saving to database (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        hard_future = pool.submit(save_hard_skills, hard_df)
        soft_future = pool.submit(save_soft_skills, soft_df)
        hard_future.result()
        soft_future.result()

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.2f} minutes.")


if __name__ == "__main__":
    main()
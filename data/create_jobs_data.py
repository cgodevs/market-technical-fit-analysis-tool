"""
ETL pipeline: LinkedIn API → enriched job postings → PostgreSQL.

Stages:
  1. Load      – read raw JSON
  2. Filter    – select columns, deduplicate
  3. Transform – rename, normalize, add metadata columns
  4. Enrich    – embed job titles → top industries; LLM seniority extraction
  5. Save      – bulk-insert into PostgreSQL
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5

DB_HOST = "localhost"
DB_NAME = "market_fit"
db_user = getenv("DB_USER")
db_pw = getenv("DB_PASSWORD")
api_key = getenv("GEMINI_API_KEY")

JOB_POSTINGS_TABLE = "job_postings"
SOURCE_FILE = "./data/linkedin_api.json"

EMBED_BATCH_SIZE = 100
EMBED_CONCURRENCY = 5
SENIORITY_CONCURRENCY = 10
INDUSTRY_CONCURRENCY = 8
TOP_INDUSTRIES = 3
EMBED_MAX_RETRIES = 3

COLUMNS_TO_KEEP = [
    "id", "date_posted", "date_created", "title", "description_text",
    "seniority", "url", "countries_derived", "locations_derived",
    "organization", "organization_logo", "linkedin_org_url",
]

SENIORITY_MAP = {
    "Pleno-sênior": "Mid",
    "Não aplicável": None,
    "Assistente": "Junior",
    "Júnior": "Junior",
    "Cadre": None,
    "Non pertinent": None,
    "Mid-Senior level": "Mid",
    "Directeur": "Director",
    "Confirmé": None,
    "Associate": "Associate",
    "Estagiário": "Intern",
    "Estágio": "Intern",
}

SENIORITY_LEVELS = (
    "Intern", "Junior", "Mid", "Senior", "Associate", "Specialist",
    "Manager", "Director", "Head", "President/Vice President",
    "C-Level", "Partner", "Owner", "Founder",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SeniorityExtraction(BaseModel):
    min_seniority: str = Field(
        description=f"Minimum seniority level. Must be one of: {', '.join(SENIORITY_LEVELS)}."
    )
    time_experience_months: Optional[float] = Field(
        description="Minimum months of experience required. Null if not mentioned."
    )

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

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


def update_job_postings_table_with_embeddings(df: pd.DataFrame) -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(f"""
                    UPDATE {JOB_POSTINGS_TABLE}
                    SET title_embedding = %s
                    WHERE id = %s
                """, (row["title_embedding"], row["id"]))
            conn.commit()


# ---------------------------------------------------------------------------
# Industry matching
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarities between a single query vector and every row of matrix."""
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    return (matrix @ query) / (row_norms * query_norm + 1e-10)


def top_k_industries(
    job_embedding: list[float],
    industry_matrix: np.ndarray,
    industry_titles: np.ndarray,
    k: int = TOP_INDUSTRIES,
) -> list[str]:
    vec = np.asarray(job_embedding, dtype=np.float32)
    sims = cosine_similarity_matrix(vec, industry_matrix)
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    upper_cased = [industry_titles[i].upper() for i in top_idx]
    return upper_cased


def assign_industries(
    titles_and_embeddings: list[tuple[str, list[float]]],
    industry_matrix: np.ndarray,
    industry_titles: np.ndarray,
    concurrency: int = INDUSTRY_CONCURRENCY,
) -> list[list[str]]:
    """Return a list of industry lists, one per input title embedding."""
    results = [None] * len(titles_and_embeddings)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(top_k_industries, emb, industry_matrix, industry_titles): i
            for i, (_, emb) in enumerate(titles_and_embeddings)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results

# ---------------------------------------------------------------------------
# Seniority extraction
# ---------------------------------------------------------------------------

def _build_seniority_chain(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are an expert technical recruiter specializing in seniority assessment.
            Extract the MINIMUM seniority level and experience duration required for the role.

            Seniority levels (lowest → highest):
            {", ".join(SENIORITY_LEVELS)}

            Rules:
            1. If seniority is already hardcoded in the input, return it as-is.
            2. Pick the MINIMUM accepted level — not the ideal candidate.
            3. If no seniority is explicit, infer from context:
            - Internship/trainee → Intern
            - Entry level / < 2 years → Junior
            - 2–4 years → Mid
            - 5+ years → Senior
            - People management → Manager or above
            4. time_experience_months: convert years to months (3 years → 36.0).
            Leave null if no duration is stated or implied.
            5. Output must use exactly one of the allowed seniority strings.
        """),
        ("human", "{input}"),
    ])
    return prompt | chat_model.with_structured_output(schema=SeniorityExtraction)


def _extract_seniority_for_row(chain, row: dict) -> tuple[str, SeniorityExtraction]:
    text = (
        f"Job title: {row['title']}\n"
        f"Job description: {row['description']}\n"
        f"Hardcoded seniority: {row['f_ai_min_seniority']}"
    )
    try:
        result = chain.invoke({"input": text})
    except Exception as exc:
        print(f"Seniority extraction failed for job {row['id']}: {exc}")
        result = SeniorityExtraction(min_seniority="Mid", time_experience_months=None)
    return row["id"], result


def extract_seniority_bulk(
    chat_model,
    df: pd.DataFrame,
    concurrency: int = SENIORITY_CONCURRENCY,
) -> pd.DataFrame:
    """Return a DataFrame with job_id, f_ai_min_seniority, ai_experience_time_months."""
    chain = _build_seniority_chain(chat_model)
    rows = df.to_dict(orient="records")

    records = [None] * len(rows)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_extract_seniority_for_row, chain, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            job_id, result = future.result()
            records[futures[future]] = {
                "job_id": job_id,
                "f_ai_min_seniority": result.min_seniority,
                "ai_experience_time_months": result.time_experience_months or 0.0,
            }

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def db_connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=db_user, password=db_pw
    )


def load_industries_from_db(conn) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings_matrix, titles_array) for all work_industries rows."""
    with conn.cursor() as cur:
        cur.execute("SELECT title, embedding FROM work_industries")
        rows = cur.fetchall()

    titles = np.array([r[0] for r in rows])
    embeddings = np.vstack([
        np.array(json.loads(r[1]) if isinstance(r[1], str) else r[1], dtype=np.float32)
        for r in rows
    ])
    return embeddings, titles


def bulk_insert_jobs(conn, df: pd.DataFrame, table: str = JOB_POSTINGS_TABLE) -> None:
    columns = [
        "id", "date_posted", "date_created", "title", "description", "url",
        "country", "location", "organization", "organization_logo",
        "linkedin_org_url", "weight", "c_source", "f_ai_min_seniority",
        "ai_experience_time_months", "ai_industries", "title_embedding"
    ]
    records = [tuple(row[c] for c in columns) for _, row in df[columns].iterrows()]

    insert_sql = f"""
        INSERT INTO {table} ({", ".join(columns)})
        VALUES %s
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, records, page_size=500)
    conn.commit()
    print(f"Inserted {len(records)} rows into {table}.")

# ---------------------------------------------------------------------------
# ETL stages
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def filter_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS_TO_KEEP].copy()
    df["weight"] = df.groupby(["organization", "title", "description_text"])["id"].transform("count")
    return df.drop_duplicates(subset=["organization", "title", "description_text"])


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "description_text": "description",
        "countries_derived": "country",
        "locations_derived": "location",
    })

    df["country"] = df["country"].apply(lambda x: x[0] if x else None)
    df["location"] = df["location"].apply(lambda x: x[0] if x else None)
    df["date_posted"] = df["date_posted"].apply(lambda x: x.split("T")[0] if x else None)
    df["date_created"] = df["date_created"].apply(lambda x: x.split("T")[0] if x else None)

    df["seniority"] = df["seniority"].map(lambda v: SENIORITY_MAP.get(v, v if v in SENIORITY_LEVELS else None))

    df["c_source"] = "LinkedIn API"
    df["f_ai_min_seniority"] = df["seniority"]
    df["ai_experience_time_months"] = None
    df["ai_industries"] = None

    return df.drop(columns=["seniority"])


def enrich(df: pd.DataFrame, gemini_client: genai.Client, chat_model) -> pd.DataFrame:
    # --- Industries via embeddings ---
    with db_connect() as conn:
        industry_matrix, industry_titles = load_industries_from_db(conn)

    title_embeddings = embed_texts(gemini_client, df["title"].tolist())
    titles_with_embeddings = list(zip(df["title"].tolist(), title_embeddings))
    industry_lists = assign_industries(titles_with_embeddings, industry_matrix, industry_titles)
    df = df.copy()
    df["ai_industries"] = industry_lists

    # --- Seniority via LLM ---
    seniority_df = extract_seniority_bulk(chat_model, df)
    df = df.merge(seniority_df, left_on="id", right_on="job_id", how="left", suffixes=("_old", ""))
    df = df.drop(columns=["job_id", "f_ai_min_seniority_old", "ai_experience_time_months_old"])

    # --- Title embeddings via Gemini ---
    df["title_embedding"] = embed_texts(gemini_client, df["title"].tolist())

    return df


def save(df: pd.DataFrame) -> None:
    with db_connect() as conn:
        bulk_insert_jobs(conn, df)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()
    chat_model = init_chat_model(
        model=MODEL_NAME,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
    )
    print("Loading data...")
    raw_df = load(SOURCE_FILE)

    print("Filtering and deduplicating...")
    filtered_df = filter_and_deduplicate(raw_df)

    print("Transforming...")
    transformed_df = transform(filtered_df)

    print("Enriching (embeddings + seniority LLM)...")
    gemini_client = genai.Client(api_key=api_key)
    enriched_df = enrich(transformed_df, gemini_client, chat_model)

    print("Saving to database...")
    save(enriched_df)

    print(f"Total processing time: {(time.time() - start_time) / 60:.2f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
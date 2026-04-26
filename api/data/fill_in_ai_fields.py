
from os import getenv
import pandas as pd
import psycopg2
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from google import genai
from typing import List


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


def embed_batch_with_retry(client: genai.Client, skills: List[str], max_retries: int = 3) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=skills
            )
            return [e.values for e in response.embeddings]
        except genai.errors.ClientError as e:
            is_rate_limit = (e.code == 429 or "RESOURCE_EXHAUSTED" in str(e.status))
            if not is_rate_limit or attempt == max_retries - 1:
                raise

            match = re.search(r'retry in (\d+(?:\.\d+)?)s', str(e))
            wait = float(match.group(1)) if match else (2 ** attempt * 10)
            print(f"Rate limited. Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)

    raise RuntimeError("Max retries exceeded")

def build_embeddings(client: genai.Client, df: pd.DataFrame, column_to_embed: str, batch_size: int = 100, concurrency: int = 5) -> pd.DataFrame:
    result_df = df.copy()
    skills = result_df[column_to_embed].tolist()

    batches = [skills[i:i + batch_size] for i in range(0, len(skills), batch_size)]

    with ThreadPool(concurrency) as pool:
        batch_results = pool.map(lambda batch: embed_batch_with_retry(client, batch), batches)

    embeddings = [emb for batch in batch_results for emb in batch]
    result_df["embedding"] = embeddings
    return result_df
    


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

    cur.execute(f"""
        SELECT 
            id, 
            title,
            embedding 
        FROM work_industries
    """)
    rows = cur.fetchall()
    industries_df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    industries_df["embedding"] = industries_df["embedding"].apply(
    lambda x: np.array(json.loads(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32)
)

jobs_df["ai_industries"] = None
jobs_df["ai_experience_time_months"] = None
jobs_titles_df = jobs_df[["id", "title"]]

client = genai.Client(api_key=api_key)
jobs_titles_df["title_embedding"] = build_embeddings(client, jobs_titles_df, "title")["embedding"]
jobs_titles_df["ai_industries"] = None  

industry_embeddings_matrix = np.vstack(industries_df["embedding"].values)
industry_titles = industries_df["title"].values

def find_top_industries_for_job_title(job_title_embedding, top_k=3):
    vec = np.array(job_title_embedding)
    # Vectorized cosine similarity against all industries at once
    norms = np.linalg.norm(industry_embeddings_matrix, axis=1) * np.linalg.norm(vec)
    similarities = industry_embeddings_matrix @ vec / norms
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    return industry_titles[top_indices].tolist()

def process_row_industry(row):
    return row.name, find_top_industries_for_job_title(row["title_embedding"])

results = {}
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_row_industry, row): row.name for _, row in jobs_titles_df.iterrows()}
    for future in as_completed(futures):
        idx, industries = future.result()
        results[idx] = industries

for idx, industries in results.items():
    jobs_titles_df.at[idx, "ai_industries"] = industries


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
    for _, row in jobs_titles_df.iterrows():
        cur.execute(f"""
            UPDATE {JOB_POSTINGS_TABLE_NAME}
            SET ai_industries = %s
            WHERE id = %s
        """, (row["ai_industries"], row["id"]))



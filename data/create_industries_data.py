import pandas as pd
import psycopg2
import time
import re
from os import getenv
from typing import List
from multiprocessing.pool import ThreadPool
from psycopg2.extras import execute_values
from google import genai

EMBEDDING_MODEL = "gemini-embedding-001"
HOST = "localhost"
DATABASE = "market_fit"
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

def get_list_of_industries_from_local_file() -> List[str]:
    with open("./data/enum_industry.txt", "r") as f:
        industries = [line.strip() for line in f if line.strip()]
    return industries


client = genai.Client(api_key=api_key)
title_list = get_list_of_industries_from_local_file()
industries_df = pd.DataFrame(title_list, columns=["title"])
industries_df_with_embeddings = build_embeddings(client, industries_df, "title")    

with psycopg2.connect(
    host=HOST,
    database=DATABASE,
    user=database_user,
    password=database_password
) as conn:
    conn.autocommit = False 

    cur = conn.cursor()
    execute_values(
        cur,
        "INSERT INTO work_industries (title, embedding) VALUES %s",
        [(row["title"], row["embedding"]) for _, row in industries_df_with_embeddings.iterrows()],
        template="(%s, %s::vector)", 
        page_size=500
    )
    conn.commit()

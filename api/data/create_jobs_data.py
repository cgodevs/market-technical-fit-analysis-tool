import json
import pandas as pd
import time
import re
import psycopg2
import numpy as np
from os import getenv
from typing import List, Optional
from pydantic import BaseModel, Field
from multiprocessing.pool import ThreadPool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model 
from google import genai
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

# =============================================================================== #
# Filter Relevant Data
# =============================================================================== #

COLUMNS_SELECTION = ["id", "date_posted", "date_created", "title", "description_text", "seniority", "url", "countries_derived", "locations_derived", "organization", "organization_logo", "linkedin_org_url"]

with open("./api/data/linkedin_api.json") as f:
    data = json.load(f)[:2] #todo remove   
    df = pd.DataFrame(data)
    selected_columns_df = df[COLUMNS_SELECTION] 

# Give more weight to job postings similar to each other and avoid cluttering database with its duplicates.
selected_columns_df["weight"] = selected_columns_df.groupby(["organization", "title", "description_text"])["id"].transform("count")    
filtered_df = selected_columns_df.drop_duplicates(subset=["organization", "title", "description_text"])

# =============================================================================== #
# Transform Data
# =============================================================================== #

renamed_columns = {
    "description_text": "description",
    "countries_derived": "country",
    "locations_derived": "location"
}
country = lambda x: x["country"][0] if x["country"] else None
location = lambda x: x["location"][0] if x["location"] else None

transformed_df = filtered_df.rename(columns=renamed_columns)

transformed_df["country"] = transformed_df.apply(country, axis=1)
transformed_df["location"] = transformed_df.apply(location, axis=1)

transformed_df["date_posted"] = transformed_df["date_posted"].apply(lambda x: x.split("T")[0] if x else None)
transformed_df["date_created"] = transformed_df["date_created"].apply(lambda x: x.split("T")[0] if x else None)

renamed_seniority = {
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
    "Estágio": "Intern"
}
transformed_df.replace({"seniority": renamed_seniority}, inplace=True)

transformed_df["c_source"] = "LinkedIn API"
transformed_df["f_ai_min_seniority"] = transformed_df["seniority"]
transformed_df["ai_experience_time_months"] = None
transformed_df["ai_industries"] = None  # List of industries extracted from job description using AI

del transformed_df["seniority"]

# =============================================================================== #
# Enrich data
# =============================================================================== #
client = genai.Client(api_key=api_key)
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
            embedding 
        FROM work_industries
    """)
    rows = cur.fetchall()
    industries_df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    industries_df["embedding"] = industries_df["embedding"].apply(
        lambda x: np.array(json.loads(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32)
    )

jobs_titles_df = transformed_df[["id", "title"]]
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

ai_enriched_df = transformed_df\
    .merge(jobs_titles_df[["id", "ai_industries"]], on="id", how="left")\
    .drop(columns=["ai_industries_x"])\
    .rename(columns={"ai_industries_y": "ai_industries"})

seniority_df = build_seniority_df(chat_model=chat_model, df=ai_enriched_df)
seniority_df["ai_experience_time_months"] = seniority_df["ai_experience_time_months"].fillna(0)

final_enriched_df = ai_enriched_df\
    .merge(seniority_df, left_on="id", right_on="job_id", how="left")\
    .drop(columns=["job_id", "f_ai_min_seniority_x", "ai_experience_time_months_x"])\
    .rename(columns={"f_ai_min_seniority_y": "f_ai_min_seniority", "ai_experience_time_months_y": "ai_experience_time_months"})

# =============================================================================== #
# Save data to local database
# =============================================================================== #

with psycopg2.connect(
    host=HOST,
    database=DATABASE,
    user=database_user,
    password=database_password
) as conn:

    if conn:
        print("Connection to the database was successful!")
        conn.autocommit = True

    cur = conn.cursor()

    for index, row in final_enriched_df.iterrows():
        insert_query = f"""
            INSERT INTO {JOB_POSTINGS_TABLE_NAME} (
                id, date_posted, date_created, title, description, url, country, location,
                organization, organization_logo, linkedin_org_url, weight, c_source,
                f_ai_min_seniority, ai_experience_time_months, ai_industries
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            --ON CONFLICT (id) DO NOTHING
        """
        values = (
            row["id"], row["date_posted"], row["date_created"], row["title"], row["description"], row["url"], row["country"], row["location"],
            row["organization"], row["organization_logo"], row["linkedin_org_url"], row["weight"], row["c_source"],
            row["f_ai_min_seniority"], row["ai_experience_time_months"], row["ai_industries"]
        )
        cur.execute(insert_query, values)

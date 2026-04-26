
import os
import re
import time
import pymupdf4llm
import psycopg2
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from pgvector.psycopg2 import register_vector
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
EMBED_BATCH_SIZE = 100
EMBED_CONCURRENCY = 5
SENIORITY_CONCURRENCY = 10
INDUSTRY_CONCURRENCY = 8
TOP_INDUSTRIES = 3
EMBED_MAX_RETRIES = 3

DB_NAME = "market_fit"
DB_HOST = "localhost"
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
api_key = os.getenv('GEMINI_API_KEY')


class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience: Optional[float] = Field(description="Experience in months, if explicitly stated.")
    weight: Optional[float] = Field(description="Relevance score 0–1 for this position.")

class SoftSkill(BaseModel):
    description: str = Field(description="Canonical name of the soft skill in English (e.g. 'Stakeholder Communication').")
    weight: Optional[float] = Field(description="Relevance score 0–1 for this position.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or experienced position")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the held position.")

class ProfessionalProfile(BaseModel):
    industries: List[str] = Field(description="Maximum of 3 matching LinkedIn industries list for the current goal job. Must be chosen from the list provided.")
    positions: List[Position] = Field(description="Maximum of 3 canonical titles for the main matching positions identified for profile description and its variations. Example: Senior Accountant, Accountant III, Accountant Specialist, Experienced Accountant. The first one is the most relevant one.")
    hard_skills: List[HardSkill]
    soft_skills: List[SoftSkill]


def db_connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    register_vector(conn)
    return conn

def get_work_industries_list() -> List[str]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM public.work_industries")
            rows = cur.fetchall()
    return [row[0] for row in rows]

def extract_professional_structured_data(text: str) -> dict:
    llm = init_chat_model(
        model=LLM_MODEL_NAME,
        model_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Your role is to extract relevant data for building a metadata object representing a professional resume. Work industries list to choose from: {'|'.join(get_work_industries_list())}"),
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


client = genai.Client(api_key=api_key)

path_my_cv = "./api/data/cv_example.pdf"
cv_md_text = pymupdf4llm.to_markdown(path_my_cv)
cv_obj = extract_professional_structured_data(cv_md_text)

industries = cv_obj["industries"]
positions = cv_obj["positions"]
#positions_embeddings = [embed_texts(client, pos.title) for pos in positions]

# TODO: Filter database by industries
# TODO: For each job, use positions embeddings for the title to calculate similarity with positions_embeddings to find the best matching positions to bring up for analysis and its time experience.
#       TODO: Maybe use a query regex filter combination of words in the title? Elasticsearch? How do search tools do it?

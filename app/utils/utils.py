import os
import re
import time
import numpy as np
import pandas as pd
import psycopg2
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from typing import List, Optional
from sklearn.cluster import DBSCAN
from contextlib import contextmanager
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

SOFT_SKILLS_SIMILARITY_THRESHOLD = 0.66
HARD_SKILLS_SIMILARITY_THRESHOLD = 0.66
SOFT_SKILLS_WEIGHT_COLUMN_INDEX = 3
SOFT_SKILLS_STRING_COLUMN_INDEX = 4
HARD_SKILLS_STRING_COLUMN_INDEX = 5
HARD_SKILLS_WEIGHT_COLUMN_INDEX = 3

LLM_MODEL_VECTOR_DIMENSIONS = 3072
DB_NAME = "market_fit"
DB_HOST = "localhost"
db_user = os.getenv("DB_USER")
db_pw = os.getenv("DB_PASSWORD")

JOB_POSTINGS_TABLE = "job_postings_general"
HARD_SKILLS_TABLE = "hard_skills"
SOFT_SKILLS_TABLE = "soft_skills"
RESUMES_TABLE = "resumes"

CANDIDATE_HARD_SKILLS_TABLE = "candidate_hard_skills"
CANDIDATE_SOFT_SKILLS_TABLE = "candidate_soft_skills"

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
EMBED_BATCH_SIZE = 100
EMBED_CONCURRENCY = 5
EMBED_MAX_RETRIES = 3

api_key = os.getenv('GEMINI_API_KEY')
SENIORITY_LEVELS = (
    "Intern", "Junior", "Mid", "Senior", "Associate", "Specialist",
    "Manager", "Director", "Head", "President/Vice President",
    "C-Level", "Partner", "Owner", "Founder",
)

class DatabaseManager:
    def __init__(self):
        self.db_config = {
            "dbname": DB_NAME,
            "host": DB_HOST,
            "user": db_user,
            "password": db_pw
        }
        self._pool = None

    @property
    def db_pool(self):
        """Lazy initialization: The pool is only created when first accessed."""
        if self._pool is None:
            print("Initializing connection pool...")
            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **self.db_config
            )
        return self._pool

    @contextmanager
    def get_conn(self):
        """Context manager to handle connection lifecycle."""
        conn = self.db_pool.getconn()
        try:
            register_vector(conn)
            yield conn
        finally:
            self.db_pool.putconn(conn)

    def get_resume(self, resume_id: str) -> pd.DataFrame:
        query = f"SELECT * FROM {RESUMES_TABLE} WHERE id = %s"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (resume_id,))
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)

    def filter_job_postings(self, industries: list) -> pd.DataFrame:
        query = f"SELECT id, title, description FROM {JOB_POSTINGS_TABLE} WHERE ai_industries::TEXT[] && %s::TEXT[]"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (industries,))
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)

    def get_position_skills(self, jobs_ids: list, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} WHERE job_id = ANY(%s)"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (jobs_ids,))
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)

    def get_candidate_skills(self, resume_id: str, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} WHERE resume_id = %s"
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (resume_id,))
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)
    
    def close_all(self):
        """Cleanly shut down the pool when the app stops."""
        if self._pool:
            self._pool.closeall()

class MarketSkillsMatrix:
    def __init__(self, database_manager: DatabaseManager, skill_type: str, candidate_industries: list):
        if skill_type not in ("hard", "soft"):
            raise ValueError(f"skill_type must be 'hard' or 'soft', got '{skill_type}'")
        if candidate_industries is None:
            raise ValueError("jobs_df cannot be empty")

        self.db = database_manager
        self.skill_type = skill_type
        self.candidate_industries = candidate_industries

        # Populated by _initialize_matrices
        self.string_matrix: np.ndarray = None      # (n_jobs, max_skills) skill descriptions
        self.weight_matrix: np.ndarray = None      # (n_jobs, max_skills) skill weights
        self.embedding_matrix: np.ndarray = None   # (n_jobs, max_skills, vector_dim) embeddings
        self.skills_count_by_index: list[int] = [] # count of skills per job index 
        self.job_id_by_index: list = []            # job_id mapped to matrix row index

        # Populated after combine() / weight_against() calls
        self._match_score_matrix: np.ndarray = None     # raw match counts per (job, skill) cell, starts with 0s and 1s
        self._weighted_match_matrix: np.ndarray = None  # weight-qualified match counts

        self._initialize_matrices()

    def _initialize_matrices(self) -> None:
        jobs_df = self.db.filter_job_postings(self.candidate_industries)
        matching_jobs_ids = jobs_df["id"].tolist()
        table = SOFT_SKILLS_TABLE if self.skill_type == "soft" else HARD_SKILLS_TABLE
        skills_df = self.db.get_position_skills(matching_jobs_ids, table).sort_values("job_id")

        skills_df["index"] = skills_df.groupby("job_id").ngroup()
        max_skills_per_job = skills_df.groupby("index").size().max()

        self.string_matrix = build_padded_matrix(
            skills_df, "skill_description", max_skills_per_job, pad_value=""
        )
        self.weight_matrix = build_padded_matrix(
            skills_df, "weight", max_skills_per_job, pad_value=0, dtype=np.float32
        )
        self.embedding_matrix = build_embedding_matrix(
            skills_df, max_skills_per_job
        )
        self.skills_count_by_index = (
            skills_df["index"].value_counts().sort_index().tolist()
        )
        self.job_id_by_index = (
            skills_df[["index", "job_id"]].drop_duplicates()["job_id"].tolist()
        )
        self._match_score_matrix = create_match_score_matrix(self.skills_count_by_index)
        self._weighted_match_matrix = np.zeros_like(self._match_score_matrix)

    def accumulate_matches(self, match_array: np.ndarray) -> None:
        """Add a match score array into the running match score matrix."""
        if match_array.shape != self._match_score_matrix.shape:
            raise ValueError(
                f"match_array shape {match_array.shape} does not match "
                f"expected {self._match_score_matrix.shape}"
            )
        self._match_score_matrix += match_array

    def accumulate_weighted_matches(
        self, candidate_skill_weight: float, binary_mask: np.ndarray
    ) -> None:
        """Record which job skills are met by a candidate skill at the given weight."""
        if binary_mask.shape != self.weight_matrix.shape:
            raise ValueError(
                f"binary_mask shape {binary_mask.shape} does not match "
                f"weight_matrix shape {self.weight_matrix.shape}"
            )
        candidate_weight_mask = binary_mask * candidate_skill_weight
        weight_qualified = (candidate_weight_mask >= self.weight_matrix) & (self.weight_matrix != 0)
        self._weighted_match_matrix += weight_qualified.astype(np.int8)

    def get_min_compliance_pct_by_job(self) -> list[float]:
        """
        Percentage of each job's skills matched by the candidate, not considering weight.
        """
        qualifying = (self._match_score_matrix > 1).sum(axis=1)  # 0 means padding and n > 1 means a candidate skill matched that job skill n times
        return [round(100 * a / b, 2) for a, b in zip(qualifying.tolist(), self.skills_count_by_index)]

    def get_ideal_compliance_pct_by_job(self) -> list[float]:
        """
        Percentage of each job's skills met at or above their required weight by candidate.
        """
        qualifying = (self._weighted_match_matrix != 0).sum(axis=1)
        return [round(100 * a / b, 2) for a, b in zip(qualifying.tolist(), self.skills_count_by_index)]

    def get_matched_skills(self, job_index: int, top_n: int = 5, with_scores: bool = False):
        scores = self._match_score_matrix[job_index]
        descriptions = self.string_matrix[job_index]
        ranked = sorted(
            ((desc, int(score)) for desc, score in zip(descriptions, scores) if desc != "" and score > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked if with_scores else [desc for desc, _ in ranked]

class HardSkill(BaseModel):
    description: str = Field(description="Objective description of the hard skill or experience in English.")
    time_experience_months: Optional[float] = Field(description="Experience in months, if explicitly stated, otherwise set to 0.")
    weight: Optional[float] = Field(description="Relevance experience score 0–1 for this position.")

class SoftSkill(BaseModel):
    description: str = Field(description="Canonical name of the soft skill in English (e.g. 'Stakeholder Communication').")
    weight: Optional[float] = Field(description="Relevance experience score 0–1 for this position.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or most experienced position")
    time_experience_months: Optional[float] = Field(description="Time experience in months identified for the held position. Set to 0 if not able to identify.")

class ProfessionalProfile(BaseModel):
    industries: List[str] = Field(description="Maximum of 2 matching LinkedIn industries list for the current goal job title, not the experience. Must be chosen from the list provided.")
    seniority: str = Field(description=f"Seniority level identifified for main goal position. Must be one of: {', '.join(SENIORITY_LEVELS)}.")
    position: Position
    hard_skills: List[HardSkill]
    soft_skills: List[SoftSkill]


def cosine_similarities_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    zero_mask = np.all(matrix == 0, axis=-1)  
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=-1)  
    dot_products = matrix @ query
    similarities = dot_products / (row_norms * query_norm + 1e-10)
    similarities[zero_mask] = 0.0
    return similarities  

def create_match_score_matrix(count_list: list):
    nrows = len(count_list)
    ncols = max(count_list)
    matrix = np.zeros((nrows, ncols), dtype=np.int8)
    for i, count in enumerate(count_list):
        matrix[i, :count] = 1
    return matrix     

def analyze_market(market_obj: MarketSkillsMatrix, candidate_skills_df: pd.DataFrame, skills_type: str) -> None:
    weight_column_index = SOFT_SKILLS_WEIGHT_COLUMN_INDEX if skills_type == "soft" else HARD_SKILLS_WEIGHT_COLUMN_INDEX
    string_column_index = SOFT_SKILLS_STRING_COLUMN_INDEX if skills_type == "soft" else HARD_SKILLS_STRING_COLUMN_INDEX
    threshold = SOFT_SKILLS_SIMILARITY_THRESHOLD if skills_type == "soft" else HARD_SKILLS_SIMILARITY_THRESHOLD
    skills_count = candidate_skills_df.shape[0]

    for i in range(0, skills_count):
        weight = candidate_skills_df.iloc[(i, weight_column_index)] 
        skill_embedding = candidate_skills_df.iloc[(i, string_column_index) ] 
        cosine_similarities = cosine_similarities_matrix(skill_embedding, market_obj.embedding_matrix)
        binary_mask = (cosine_similarities > threshold).astype(np.int8)
        market_obj.accumulate_matches(binary_mask)
        market_obj.accumulate_weighted_matches(weight, binary_mask)

def build_padded_matrix(
        df: pd.DataFrame,
        column_name: str,
        length: int,
        pad_value,
        dtype=None,
    ) -> np.ndarray:
        rows = [
            np.pad(
                array=group[column_name].values,
                pad_width=(0, length - len(group)),
                constant_values=pad_value,
            )
            for _, group in df.groupby("index")
        ]
        return np.array(rows, dtype=dtype)

def build_embedding_matrix(
    df: pd.DataFrame, max_skills: int
    ) -> np.ndarray:
        zero_vector = np.zeros(LLM_MODEL_VECTOR_DIMENSIONS)
        rows = [
            np.vstack(
                list(group["embedding"].values)
                + [zero_vector] * (max_skills - len(group))
            )
            for _, group in df.groupby("index")
        ]
        return np.array(rows, dtype=np.float32)

def get_market_analysis_results(market_obj: MarketSkillsMatrix) -> list[dict]:
    compliance_by_job = market_obj.get_min_compliance_pct_by_job()
    ideal_compliance_by_job = market_obj.get_ideal_compliance_pct_by_job()
    noncompliance_mask = market_obj._match_score_matrix == 1

    return [
        {
            "job_index": i,
            "job_id": job_id,
            "minimum_compliance_pct": compliance_pct,
            "ideal_compliance_pct": ideal_compliance_pct,

            "nonmatched_skills_count": int(noncompliance_mask[i].sum()),
            "nonmatched_skills": list(set(market_obj.string_matrix[i][noncompliance_mask[i]])),
            
            "matched_skills": list(
                market_obj.string_matrix[i][
                    (market_obj._match_score_matrix[i] > 1) &
                    (market_obj.string_matrix[i] != "")
                ]
            ),
            "similarity_match_scores": market_obj.get_matched_skills(i, with_scores=True),

            "not_ideal_skills": list(set(
                    market_obj.string_matrix[i][
                        (market_obj._weighted_match_matrix[i] == 0) &
                        (market_obj.string_matrix[i] != "")
                    ]
                ))
        }
        for i, (job_id, compliance_pct, ideal_compliance_pct) in enumerate(
            zip(market_obj.job_id_by_index, compliance_by_job, ideal_compliance_by_job)
        )
    ]

def build_analysis_display(market_obj: MarketSkillsMatrix, analysis: list[dict]) -> pd.DataFrame:
    sorted_analysis = sorted(analysis, key=lambda e: e["minimum_compliance_pct"], reverse=True)
    sorted_counts = [market_obj.skills_count_by_index[market_obj.job_id_by_index.index(e["job_id"])] for e in sorted_analysis]

    df = pd.DataFrame([
        {
            "job_id": entry["job_id"],
            "job_index": entry["job_index"],
            "required_skills": count,
            "matched_count": len(entry["matched_skills"]),
            "minimum_compliance_pct": entry["minimum_compliance_pct"],
            "matched_skills": entry["matched_skills"],

            "insufficient_count": len(entry["not_ideal_skills"]),
            "ideal_compliance_pct": entry["ideal_compliance_pct"],
            "insufficient_proficiency": entry["not_ideal_skills"],

            "nonmatched_count": entry["nonmatched_skills_count"],
            "nonmatched_skills": entry["nonmatched_skills"],
        }
        for entry, count in zip(sorted_analysis, sorted_counts)
    ])
    return df

def matches_display(market_obj: MarketSkillsMatrix, analysis: list[dict]) -> pd.DataFrame:
    flat_data = [
        (skill, score, entry["job_index"])
        for entry in analysis
        for skill, score in entry["similarity_match_scores"]
    ]
    if not flat_data:
        return pd.DataFrame()

    matches_df = pd.DataFrame(flat_data, columns=["skill", "score", "job_index"])

    skill_stats = matches_df.groupby("skill").agg(
        total_matches=("score", "sum"),
        job_indices=("job_index", set)
    ).reset_index()
    flat_strings = market_obj.string_matrix.flatten()
    flat_embeddings = market_obj.embedding_matrix.reshape(-1, LLM_MODEL_VECTOR_DIMENSIONS)
    
    skill_stats["embedding"] = [
        flat_embeddings[np.where(flat_strings == skill)[0][0]]
        if np.where(flat_strings == skill)[0].size > 0 else None
        for skill in skill_stats["skill"]
    ]
    skill_stats = skill_stats.dropna(subset=["embedding"])

    embeddings = np.vstack(skill_stats["embedding"].values)
    skill_stats["cluster"] = DBSCAN(eps=0.15, min_samples=1, metric="cosine").fit_predict(embeddings)

    total_jobs = len(analysis)
    df = (
        skill_stats
        .groupby("cluster")
        .agg(
            skill_variants=("skill", list),
            total_matches=("total_matches", "sum"),
            unique_jobs=("job_indices", lambda sets: len(set.union(*sets)))
        )
        .assign(job_coverage_pct=lambda df: (df["unique_jobs"] / total_jobs * 100).round(2))
        .sort_values("total_matches", ascending=False)
        .reset_index(drop=True)  # drops the cluster index cleanly
    )
    return df[df["skill_variants"].apply(len) < 10] # Avoids poorly formed skill variants where a large number of non related items are grouped together

def nonmatches_display(market_obj: MarketSkillsMatrix) -> pd.DataFrame:
    nonmatch_mask = market_obj._match_score_matrix == 1
    flat_embeddings = market_obj.embedding_matrix[nonmatch_mask]
    flat_descriptions = market_obj.string_matrix[nonmatch_mask]

    if len(flat_embeddings) == 0:
        return pd.DataFrame()

    # Recover which job each nonmatched skill belongs to
    job_indices = np.where(nonmatch_mask)[0]  # row index = job index

    cluster_labels = DBSCAN(eps=0.2, min_samples=2, metric="cosine").fit_predict(flat_embeddings)

    total_jobs = len(market_obj.job_id_by_index)
    df = (
        pd.DataFrame({"skill": flat_descriptions, "job_index": job_indices, "cluster": cluster_labels})
        .groupby("cluster")
        .agg(
            skill_variants=("skill", lambda x: list(set(x))),
            total_matches=("skill", "count"),
            unique_jobs=("job_index", "nunique")
        )
        .assign(job_coverage_pct=lambda df: (df["unique_jobs"] / total_jobs * 100).round(2))
        .sort_values("total_matches", ascending=False)
        .reset_index(drop=True)
    )
    return df[df["skill_variants"].apply(len) < 10] # Avoids poorly formed skill variants where a large number of non related items are grouped together

# ============== Extract data from resume and build structured professional profile ==============
def get_list_of_industries_from_local_file() -> List[str]:
    with open("./enum_industry.txt", "r") as f:
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
        conn.commit()

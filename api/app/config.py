from os import getenv

SOFT_SKILLS_SIMILARITY_THRESHOLD = 0.66
HARD_SKILLS_SIMILARITY_THRESHOLD = 0.66
SOFT_SKILLS_WEIGHT_COLUMN_INDEX = 3
SOFT_SKILLS_STRING_COLUMN_INDEX = 4
HARD_SKILLS_STRING_COLUMN_INDEX = 5
HARD_SKILLS_WEIGHT_COLUMN_INDEX = 3

JOB_POSTINGS_TABLE = "job_postings_general"
HARD_SKILLS_TABLE = "hard_skills"
SOFT_SKILLS_TABLE = "soft_skills"
RESUMES_TABLE = "resumes"
CANDIDATE_HARD_SKILLS_TABLE = "candidate_hard_skills"
CANDIDATE_SOFT_SKILLS_TABLE = "candidate_soft_skills"

LLM_MODEL_VECTOR_DIMENSIONS = 3072
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
EMBED_BATCH_SIZE = 100
EMBED_CONCURRENCY = 5
EMBED_MAX_RETRIES = 3

DB_NAME = "market_fit"
DB_HOST = "host.docker.internal"  # "localhost" for local development with "fastapi dev", host.docker.internal for docker
db_user = getenv("DB_USER")
db_pw = getenv("DB_PASSWORD")
api_key = getenv('GEMINI_API_KEY')

SENIORITY_LEVELS = (
    "Intern", "Junior", "Mid", "Senior", "Associate", "Specialist",
    "Manager", "Director", "Head", "President/Vice President",
    "C-Level", "Partner", "Owner", "Founder",
)
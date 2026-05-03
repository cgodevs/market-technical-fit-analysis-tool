from os import getenv

DB_NAME = "market_fit"
DB_HOST = "localhost"
db_user = getenv("DB_USER")
db_pw = getenv("DB_PASSWORD")

RESUMES_TABLE = "resumes"
CANDIDATE_HARD_SKILLS_TABLE = "candidate_hard_skills"
CANDIDATE_SOFT_SKILLS_TABLE = "candidate_soft_skills"

SENIORITY_LEVELS = (
    "Intern", "Junior", "Mid", "Senior", "Associate", "Specialist",
    "Manager", "Director", "Head", "President/Vice President",
    "C-Level", "Partner", "Owner", "Founder",
)
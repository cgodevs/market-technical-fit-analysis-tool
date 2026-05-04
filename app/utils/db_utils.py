from typing import List
from config import RESUMES_TABLE, CANDIDATE_HARD_SKILLS_TABLE, CANDIDATE_SOFT_SKILLS_TABLE
from psycopg2.extras import execute_values
from psycopg2.extensions import connection as PsycopgConnection
import pandas as pd

def _safe_scalar(value):
    """Convert a single value to a safe type for database insertion."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return 0
    if isinstance(value, int) and not isinstance(value, bool):
        if value < -32768 or value > 32767:
            return 0
    return value

def _insert_df(conn: PsycopgConnection, df: pd.DataFrame, table_name: str) -> None:
    """Low-level insert — expects an already-open connection, no commit."""
    cols = ", ".join(df.columns)
    placeholders = ", ".join(
        "%s::vector" if "embedding" in col else "%s"
        for col in df.columns
    )
    template = f"({placeholders})"
    cur = conn.cursor()
    execute_values(
        cur,
        f"INSERT INTO {table_name} ({cols}) VALUES %s",
        [tuple(_safe_scalar(x) for x in row) for _, row in df.iterrows()],
        template=template,
        page_size=500
    )

def get_static_list_of_industries() -> List[str]:
    """Load a static list of industries from a text file."""
    with open("../data/enum_industry.txt", "r") as f:
        industries = [line.strip() for line in f if line.strip()]
        industries = [industry.upper() for industry in industries]
    return industries

def save_resume_data(
    conn: PsycopgConnection,
    cv_df: pd.DataFrame,
    hard_skills_df: pd.DataFrame,
    soft_skills_df: pd.DataFrame,
) -> None:
    """Insert resume + skills atomically — all succeed or all roll back."""
    try:
        _insert_df(conn, cv_df, RESUMES_TABLE)
        print(f"Inserted to {RESUMES_TABLE}")
        _insert_df(conn, hard_skills_df, CANDIDATE_HARD_SKILLS_TABLE)
        print(f"Inserted to {CANDIDATE_HARD_SKILLS_TABLE}")
        _insert_df(conn, soft_skills_df, CANDIDATE_SOFT_SKILLS_TABLE)
        print(f"Inserted to {CANDIDATE_SOFT_SKILLS_TABLE}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

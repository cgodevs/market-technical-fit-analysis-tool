from typing import List
from ..config import DB_HOST, DB_NAME, db_user, db_pw, RESUMES_TABLE, CANDIDATE_HARD_SKILLS_TABLE, CANDIDATE_SOFT_SKILLS_TABLE
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
    
def get_static_list_of_industries() -> List[str]:
    with open("../data/enum_industry.txt", "r") as f:
        industries = [line.strip() for line in f if line.strip()]
        industries = [industry.upper() for industry in industries] 
    return industries

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

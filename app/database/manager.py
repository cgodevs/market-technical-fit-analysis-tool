import pandas as pd
from ..config import *
from contextlib import contextmanager
from pgvector.psycopg2 import register_vector
from psycopg2 import pool


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
            self._pool = pool.SimpleConnectionPool(
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

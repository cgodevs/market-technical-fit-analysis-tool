import pandas as pd
import numpy as np
from api.app.config import LLM_MODEL_VECTOR_DIMENSIONS


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

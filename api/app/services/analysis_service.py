import numpy as np
import pandas as pd
from threading import Lock
from cachetools import TTLCache, cached
from sklearn.cluster import DBSCAN
from api.app.exceptions import ResumeNotFoundError, ResumeProcessingError, DatabaseConnectionError, DatabaseQueryError
from api.app.database.manager import DatabaseManager
from psycopg2 import OperationalError, Error
from api.app.models.responses import SkillClusterSchema, SkillsCoverageResponse, AnalysisDisplayResponse
from api.app.utils.matrix import (
    cosine_similarities_matrix, create_match_score_matrix, build_embedding_matrix, build_padded_matrix
)
from api.app.config import (
    SOFT_SKILLS_TABLE, HARD_SKILLS_TABLE, SOFT_SKILLS_SIMILARITY_THRESHOLD, 
    HARD_SKILLS_SIMILARITY_THRESHOLD, LLM_MODEL_VECTOR_DIMENSIONS, SOFT_SKILLS_STRING_COLUMN_INDEX,
    SOFT_SKILLS_WEIGHT_COLUMN_INDEX, HARD_SKILLS_STRING_COLUMN_INDEX,HARD_SKILLS_WEIGHT_COLUMN_INDEX,
    CANDIDATE_HARD_SKILLS_TABLE, CANDIDATE_SOFT_SKILLS_TABLE
)

_cache = TTLCache(maxsize=64, ttl=300)  # 5 minutes
_lock = Lock()
_market_cache = TTLCache(maxsize=32, ttl=600)  # 10 minutes
_market_lock = Lock()


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

    def get_matched_skills(self, job_index: int, with_scores: bool = False):
        scores = self._match_score_matrix[job_index]
        descriptions = self.string_matrix[job_index]
        ranked = sorted(
            ((desc, int(score)) for desc, score in zip(descriptions, scores) if desc != "" and score > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked if with_scores else [desc for desc, _ in ranked]


@cached(_cache, lock=_lock)
def _cached_analysis(resume_id: str, skill_type: str) -> list[AnalysisDisplayResponse]:
    db = DatabaseManager()
    try:
        return build_analysis_display(db, resume_id, skill_type)
    finally:
        db.close_all()

@cached(_market_cache, key=lambda db, resume_id: resume_id, lock=_market_lock)
def _analyze_market_for_coverage(db: DatabaseManager, resume_id: str) -> tuple[MarketSkillsMatrix, MarketSkillsMatrix]:
    try:
        resume_df = db.get_resume(resume_id)
    except OperationalError as e:
        raise DatabaseConnectionError(detail="Could not connect to database") from e
    except Error as e:
        raise DatabaseQueryError(detail="Failed to fetch resume") from e

    if resume_df.empty:
        raise ResumeNotFoundError(resume_id=resume_id)

    try:
        candidate_industries = resume_df["industries"][0]
    except (KeyError, IndexError) as e:
        raise ResumeProcessingError(detail="Resume is missing 'industries' field") from e

    try:
        candidate_hard_skills_df = db.get_candidate_skills(resume_id, CANDIDATE_HARD_SKILLS_TABLE)
        candidate_soft_skills_df = db.get_candidate_skills(resume_id, CANDIDATE_SOFT_SKILLS_TABLE)
    except OperationalError as e:
        raise DatabaseConnectionError(detail="Could not connect to database") from e
    except Error as e:
        raise DatabaseQueryError(detail="Failed to fetch candidate skills") from e

    if candidate_hard_skills_df.empty or candidate_soft_skills_df.empty:
        raise ResumeProcessingError(detail=f"No skills found for resume {resume_id}")

    try:
        soft_market = MarketSkillsMatrix(database_manager=db, skill_type="soft", candidate_industries=candidate_industries)
        hard_market = MarketSkillsMatrix(database_manager=db, skill_type="hard", candidate_industries=candidate_industries)
    except Error as e:
        raise DatabaseQueryError(detail="Failed to fetch market skills data") from e
    except ValueError as e:
        raise ResumeProcessingError(detail=str(e)) from e

    try:
        _analyze_market(soft_market, candidate_soft_skills_df, "soft")
        _analyze_market(hard_market, candidate_hard_skills_df, "hard")
    except ValueError as e:
        raise ResumeProcessingError(detail=f"Market analysis failed: {str(e)}") from e
    except Exception as e:
        raise ResumeProcessingError(detail="Unexpected error during market analysis") from e

    return hard_market, soft_market

def _analyze_market(market_obj: MarketSkillsMatrix, candidate_skills_df: pd.DataFrame, skills_type: str) -> None:
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

def _get_market_analysis_results(market_obj: MarketSkillsMatrix) -> list[dict]:
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

def _matches_display(market_obj: MarketSkillsMatrix, analysis: list[dict]) -> pd.DataFrame:
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

def _nonmatches_display(market_obj: MarketSkillsMatrix) -> pd.DataFrame:
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

def get_compliant_skills_coverage(db: DatabaseManager, resume_id: str) -> SkillsCoverageResponse:
    hard_market, soft_market = _analyze_market_for_coverage(db, resume_id)
    market_soft_skills_analysis = _get_market_analysis_results(soft_market)
    market_hard_skills_analysis = _get_market_analysis_results(hard_market)
    compliant_soft_skills_report = _matches_display(soft_market, market_soft_skills_analysis)    
    compliant_hard_skills_report = _matches_display(hard_market, market_hard_skills_analysis) 
    return SkillsCoverageResponse(
        soft_skills=[SkillClusterSchema(**row) for row in compliant_soft_skills_report.to_dict(orient="records")],
        hard_skills=[SkillClusterSchema(**row) for row in compliant_hard_skills_report.to_dict(orient="records")]
    )

def get_noncompliant_skills_coverage(db: DatabaseManager, resume_id: str) -> SkillsCoverageResponse:
    hard_market, soft_market = _analyze_market_for_coverage(db, resume_id)
    noncompliant_soft_skills_report = _nonmatches_display(soft_market)
    noncompliant_hard_skills_report = _nonmatches_display(hard_market)
    return SkillsCoverageResponse(
        soft_skills=[SkillClusterSchema(**row) for row in noncompliant_soft_skills_report.to_dict(orient="records")],
        hard_skills=[SkillClusterSchema(**row) for row in noncompliant_hard_skills_report.to_dict(orient="records")]
    )

def build_analysis_display(db: DatabaseManager, resume_id: str, skill_type: str) -> list[AnalysisDisplayResponse]:
    hard_market, soft_market = _analyze_market_for_coverage(db, resume_id)
    if skill_type == "soft":
        market_obj = soft_market
        analysis = _get_market_analysis_results(market_obj)
    elif skill_type == "hard":
        market_obj = hard_market
        analysis = _get_market_analysis_results(market_obj)
    else:
        raise ValueError(f"Invalid skill_type '{skill_type}', expected 'hard' or 'soft'")

    sorted_analysis = sorted(analysis, key=lambda e: e["minimum_compliance_pct"], reverse=True)
    sorted_counts = [market_obj.skills_count_by_index[market_obj.job_id_by_index.index(e["job_id"])] for e in sorted_analysis]
    rows = [
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
    ]
    return [AnalysisDisplayResponse(**row) for row in rows]

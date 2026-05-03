from analysis_utils import *

RESUME_ID = "ed8a492e-d72b-488d-924a-e198c027aa79"  # REMOVE, must be a parameter

db = DatabaseManager()

resume_df = db.get_resume(RESUME_ID)
candidate_industries = resume_df["industries"][0]

candidate_hard_skills_df = db.get_candidate_skills(RESUME_ID, CANDIDATE_HARD_SKILLS_TABLE)
candidate_soft_skills_df = db.get_candidate_skills(RESUME_ID, CANDIDATE_SOFT_SKILLS_TABLE)

soft_market = MarketSkillsMatrix(database_manager=db, skill_type="soft", candidate_industries=candidate_industries)
hard_market = MarketSkillsMatrix(database_manager=db, skill_type="hard", candidate_industries=candidate_industries)

analyze_market(soft_market, candidate_soft_skills_df, "soft")
analyze_market(hard_market, candidate_hard_skills_df, "hard")

db.close_all()

noncompliant_soft_skills_report = nonmatches_display(soft_market)
noncompliant_hard_skills_report = nonmatches_display(hard_market)

noncompliant_skills_coverage = { 
    "soft_skills": noncompliant_soft_skills_report.to_dict(orient="records"),
    "hard_skills": noncompliant_hard_skills_report.to_dict(orient="records") 
}
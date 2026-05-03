from fastapi import FastAPI
from analysis_utils import DatabaseManager

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/resumes/{resume_id}")
async def candidate_resume(resume_id: str):
    db = DatabaseManager()
    resume_df = db.get_resume(resume_id)
    resume_obj = resume_df.to_dict(orient="records")
    for record in resume_obj:
        if "position_embedding" in record:
            record["position_embedding"] = record["position_embedding"].tolist()
    db.close_all()
    return resume_obj
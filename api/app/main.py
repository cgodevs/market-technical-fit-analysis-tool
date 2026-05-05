from fastapi import FastAPI
from api.app.routers import resumes
from api.app.routers import analysis

app = FastAPI()
app.include_router(resumes.router)
app.include_router(analysis.router)


@app.get("/")
def root():
    return {"message": "Hello World"}


from fastapi import FastAPI
from routers import resumes
from routers import analysis

app = FastAPI()
app.include_router(resumes.router)
app.include_router(analysis.router)


@app.get("/")
def root():
    return {"message": "Hello World"}

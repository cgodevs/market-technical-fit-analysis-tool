from fastapi import FastAPI
from .routers import resumes

app = FastAPI()
app.include_router(resumes.router)


@app.get("/")
def root():
    return {"message": "Hello World"}


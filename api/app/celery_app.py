from celery import Celery

celery = Celery(
    "resume_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

# Autodiscover tasks from services module
celery.autodiscover_tasks(['services'])
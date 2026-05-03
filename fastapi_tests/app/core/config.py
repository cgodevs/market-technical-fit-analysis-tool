from os import getenv

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_PROVIDER = "google_genai"
LLM_TEMPERATURE = 0.5
EMBED_BATCH_SIZE = 100
EMBED_CONCURRENCY = 5
EMBED_MAX_RETRIES = 3
api_key = getenv('GEMINI_API_KEY')

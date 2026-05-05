class ResumeParsingError(Exception):
    def __init__(self, detail: str = "Failed transforming file content into markdown to parse resume"):
        self.detail = detail
        super().__init__(detail)

class ResumeNotFoundError(Exception):
    def __init__(self, resume_id: str):
        self.detail = f"Resume with id {resume_id} not found"
        super().__init__(self.detail)

class ResumeProcessingError(Exception):
    def __init__(self, detail: str = "Failed to process resume"):
        self.detail = detail
        super().__init__(detail)

class DatabaseSaveError(Exception):
    def __init__(self, table: str, detail: str = ""):
        self.detail = f"Failed to save to {table}: {detail}"
        super().__init__(self.detail)

class StructuredOutputParsingError(Exception):
    def __init__(self, detail: str = "Failed to parse structured output from LLM"):
        self.detail = detail
        super().__init__(detail)

class EmbeddingError(Exception):
    def __init__(self, detail: str = "Failed to embed text"):
        self.detail = detail
        super().__init__(detail)

class DatabaseConnectionError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)

class DatabaseQueryError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)
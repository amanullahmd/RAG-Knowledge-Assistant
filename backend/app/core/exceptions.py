"""Backend core module - configuration and exceptions"""

from fastapi import HTTPException, status

class DocumentNotFoundError(HTTPException):
    def __init__(self, doc_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found"
        )

class EmbeddingError(HTTPException):
    def __init__(self, message: str = "Failed to generate embeddings"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )

class DocumentProcessingError(HTTPException):
    def __init__(self, message: str = "Failed to process document"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message
        )

class RetrievalError(HTTPException):
    def __init__(self, message: str = "Failed to retrieve documents"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )

"""Tests for custom exceptions"""

import pytest
from fastapi import HTTPException
from backend.app.core.exceptions import (
    DocumentNotFoundError,
    EmbeddingError,
    DocumentProcessingError,
    RetrievalError,
)


class TestExceptions:
    """Test custom exceptions"""

    def test_document_not_found(self):
        exc = DocumentNotFoundError("doc-123")
        assert exc.status_code == 404
        assert "doc-123" in exc.detail

    def test_embedding_error(self):
        exc = EmbeddingError("API failed")
        assert exc.status_code == 500
        assert "API failed" in exc.detail

    def test_embedding_error_default(self):
        exc = EmbeddingError()
        assert exc.status_code == 500
        assert "embeddings" in exc.detail.lower()

    def test_document_processing_error(self):
        exc = DocumentProcessingError("Invalid format")
        assert exc.status_code == 422
        assert "Invalid format" in exc.detail

    def test_retrieval_error(self):
        exc = RetrievalError("Search failed")
        assert exc.status_code == 500
        assert "Search failed" in exc.detail

    def test_all_inherit_http_exception(self):
        """All custom exceptions should inherit from HTTPException"""
        assert issubclass(DocumentNotFoundError, HTTPException)
        assert issubclass(EmbeddingError, HTTPException)
        assert issubclass(DocumentProcessingError, HTTPException)
        assert issubclass(RetrievalError, HTTPException)

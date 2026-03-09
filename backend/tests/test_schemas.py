"""Tests for Pydantic schemas"""

import pytest
from datetime import datetime
from backend.app.models.schemas import (
    DocumentResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatMessage,
    Citation,
    ChunkResponse,
    ChatHistoryResponse,
)


class TestDocumentResponse:
    """Test DocumentResponse schema"""

    def test_valid_document(self):
        doc = DocumentResponse(
            doc_id="abc-123",
            filename="test.pdf",
            file_type="application/pdf",
            size_bytes=1024,
            uploaded_at=datetime.now(),
            chunks_count=5,
        )
        assert doc.doc_id == "abc-123"
        assert doc.filename == "test.pdf"
        assert doc.chunks_count == 5

    def test_document_serialization(self):
        doc = DocumentResponse(
            doc_id="abc-123",
            filename="test.pdf",
            file_type="application/pdf",
            size_bytes=1024,
            uploaded_at=datetime.now(),
            chunks_count=5,
        )
        data = doc.model_dump()
        assert "doc_id" in data
        assert "filename" in data


class TestChatQueryRequest:
    """Test ChatQueryRequest schema"""

    def test_minimal_request(self):
        req = ChatQueryRequest(query="What is AI?")
        assert req.query == "What is AI?"
        assert req.session_id is None
        assert req.top_k == 5

    def test_full_request(self):
        req = ChatQueryRequest(
            query="What is AI?",
            session_id="session-123",
            top_k=10,
        )
        assert req.session_id == "session-123"
        assert req.top_k == 10


class TestCitation:
    """Test Citation schema"""

    def test_full_citation(self):
        citation = Citation(
            source="doc.pdf",
            page=3,
            section="Introduction",
            content_snippet="AI is transforming...",
        )
        assert citation.source == "doc.pdf"
        assert citation.page == 3

    def test_minimal_citation(self):
        citation = Citation(
            source="doc.txt",
            content_snippet="Some content",
        )
        assert citation.page is None
        assert citation.section is None


class TestChatQueryResponse:
    """Test ChatQueryResponse schema"""

    def test_valid_response(self):
        resp = ChatQueryResponse(
            answer="AI is artificial intelligence.",
            citations=[
                Citation(source="doc.pdf", content_snippet="AI is...")
            ],
            session_id="session-123",
            timestamp=datetime.now(),
        )
        assert resp.answer == "AI is artificial intelligence."
        assert len(resp.citations) == 1

    def test_empty_citations(self):
        resp = ChatQueryResponse(
            answer="No information found.",
            citations=[],
            session_id="session-123",
            timestamp=datetime.now(),
        )
        assert resp.citations == []


class TestChatMessage:
    """Test ChatMessage schema"""

    def test_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.timestamp is None

    def test_assistant_message_with_timestamp(self):
        now = datetime.now()
        msg = ChatMessage(role="assistant", content="Hi!", timestamp=now)
        assert msg.timestamp == now


class TestChunkResponse:
    """Test ChunkResponse schema"""

    def test_chunk(self):
        chunk = ChunkResponse(
            chunk_id="doc1_0",
            doc_id="doc1",
            content="Some text content",
            page_number=1,
            metadata={"source": "test.pdf"},
        )
        assert chunk.chunk_id == "doc1_0"
        assert chunk.page_number == 1

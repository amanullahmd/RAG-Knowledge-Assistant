"""Tests for vector store service"""

import pytest
from backend.app.services.vector_store import VectorStore


class TestVectorStore:
    """Test suite for VectorStore"""

    def test_init(self, vector_store):
        """Test VectorStore initialization"""
        assert vector_store.collection is not None
        assert vector_store.count() == 0

    def test_add_documents(self, vector_store, sample_chunks, sample_embeddings, sample_metadata):
        """Test adding documents to vector store"""
        chunk_ids = vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc_1",
            metadata=sample_metadata,
        )
        assert len(chunk_ids) == len(sample_chunks)
        assert vector_store.count() == len(sample_chunks)

    def test_add_documents_empty(self, vector_store):
        """Test adding empty documents"""
        result = vector_store.add_documents(
            chunks=[], embeddings=[], doc_id="empty_doc"
        )
        assert result == []
        assert vector_store.count() == 0

    def test_add_documents_auto_metadata(self, vector_store, sample_chunks, sample_embeddings):
        """Test that metadata is auto-generated with doc_id"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="auto_meta_doc",
        )
        results = vector_store.get_by_doc_id("auto_meta_doc")
        assert len(results) == len(sample_chunks)
        for r in results:
            assert r["metadata"]["doc_id"] == "auto_meta_doc"

    def test_search(self, vector_store, sample_chunks, sample_embeddings, sample_metadata):
        """Test searching vector store"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="search_doc",
            metadata=sample_metadata,
        )
        results = vector_store.search(sample_embeddings[0], top_k=3)
        assert len(results) <= 3
        assert results[0]["content"] in sample_chunks
        assert "distance" in results[0]
        assert "metadata" in results[0]

    def test_search_returns_correct_fields(self, vector_store, sample_chunks, sample_embeddings):
        """Test search result structure"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="fields_doc",
        )
        results = vector_store.search(sample_embeddings[0], top_k=1)
        assert len(results) == 1
        result = results[0]
        assert "chunk_id" in result
        assert "content" in result
        assert "distance" in result
        assert "metadata" in result

    def test_search_top_k_limit(self, vector_store, sample_chunks, sample_embeddings):
        """Test top_k limits results"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="limit_doc",
        )
        results = vector_store.search(sample_embeddings[0], top_k=2)
        assert len(results) == 2

    def test_get_by_doc_id(self, vector_store, sample_chunks, sample_embeddings, sample_metadata):
        """Test retrieving chunks by document ID"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="get_doc",
            metadata=sample_metadata,
        )
        results = vector_store.get_by_doc_id("get_doc")
        assert len(results) == len(sample_chunks)

    def test_get_by_doc_id_not_found(self, vector_store):
        """Test retrieving non-existent document"""
        results = vector_store.get_by_doc_id("nonexistent")
        assert results == []

    def test_delete_document(self, vector_store, sample_chunks, sample_embeddings, sample_metadata):
        """Test deleting a document"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="delete_doc",
            metadata=sample_metadata,
        )
        assert vector_store.count() == len(sample_chunks)

        result = vector_store.delete_document("delete_doc")
        assert result is True
        assert vector_store.count() == 0

    def test_delete_nonexistent_document(self, vector_store):
        """Test deleting non-existent document returns True (no-op)"""
        result = vector_store.delete_document("nonexistent")
        assert result is True

    def test_clear(self, vector_store, sample_chunks, sample_embeddings):
        """Test clearing all documents"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="clear_doc",
        )
        assert vector_store.count() > 0

        vector_store.clear()
        assert vector_store.count() == 0

    def test_multiple_documents(self, vector_store, sample_embeddings):
        """Test adding and querying multiple documents"""
        chunks_1 = ["Doc 1 chunk 1", "Doc 1 chunk 2"]
        chunks_2 = ["Doc 2 chunk 1", "Doc 2 chunk 2", "Doc 2 chunk 3"]

        vector_store.add_documents(
            chunks=chunks_1,
            embeddings=sample_embeddings[:2],
            doc_id="doc_1",
        )
        vector_store.add_documents(
            chunks=chunks_2,
            embeddings=sample_embeddings[:3],
            doc_id="doc_2",
        )

        assert vector_store.count() == 5
        assert len(vector_store.get_by_doc_id("doc_1")) == 2
        assert len(vector_store.get_by_doc_id("doc_2")) == 3

    def test_get_all_documents(self, vector_store, sample_chunks, sample_embeddings):
        """Test getting all documents"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="all_doc",
        )
        results = vector_store.get_all_documents()
        assert len(results["ids"]) == len(sample_chunks)
        assert len(results["documents"]) == len(sample_chunks)

    def test_count(self, vector_store, sample_chunks, sample_embeddings):
        """Test count method"""
        assert vector_store.count() == 0
        vector_store.add_documents(
            chunks=sample_chunks[:2],
            embeddings=sample_embeddings[:2],
            doc_id="count_doc",
        )
        assert vector_store.count() == 2

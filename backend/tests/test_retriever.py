"""Tests for hybrid retriever"""

import pytest
from backend.app.services.retriever import HybridRetriever
from backend.app.services.vector_store import VectorStore


class TestHybridRetriever:
    """Test suite for HybridRetriever"""

    def test_init_empty_store(self, vector_store):
        """Test initializing retriever with empty vector store"""
        retriever = HybridRetriever(vector_store)
        assert retriever.bm25_index is None
        assert retriever.doc_texts == []

    def test_init_with_data(self, vector_store, sample_chunks, sample_embeddings):
        """Test initializing retriever with data"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="init_doc",
        )
        retriever = HybridRetriever(vector_store)
        assert retriever.bm25_index is not None
        assert len(retriever.doc_texts) == len(sample_chunks)

    def test_rebuild_index(self, vector_store, sample_chunks, sample_embeddings):
        """Test rebuilding BM25 index"""
        retriever = HybridRetriever(vector_store)
        assert retriever.bm25_index is None

        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="rebuild_doc",
        )
        retriever.rebuild_index()
        assert retriever.bm25_index is not None
        assert len(retriever.doc_texts) == len(sample_chunks)

    def test_retrieve_returns_results(self, vector_store, sample_chunks, sample_embeddings):
        """Test retrieval returns results"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="retrieve_doc",
        )
        retriever = HybridRetriever(vector_store)

        results = retriever.retrieve(
            query="artificial intelligence",
            query_embedding=sample_embeddings[0],
            top_k=3,
        )
        assert len(results) <= 3
        assert len(results) > 0

    def test_retrieve_result_structure(self, vector_store, sample_chunks, sample_embeddings):
        """Test retrieval result has expected fields"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="structure_doc",
        )
        retriever = HybridRetriever(vector_store)

        results = retriever.retrieve(
            query="test query",
            query_embedding=sample_embeddings[0],
            top_k=1,
        )
        assert len(results) > 0
        result = results[0]
        assert "chunk_id" in result
        assert "content" in result
        assert "rrf_score" in result

    def test_retrieve_empty_store(self, vector_store, sample_embeddings):
        """Test retrieval from empty store"""
        retriever = HybridRetriever(vector_store)
        results = retriever.retrieve(
            query="test",
            query_embedding=sample_embeddings[0],
            top_k=5,
        )
        assert results == []

    def test_rrf_scores_are_positive(self, vector_store, sample_chunks, sample_embeddings):
        """Test that RRF scores are positive"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="rrf_doc",
        )
        retriever = HybridRetriever(vector_store)

        results = retriever.retrieve(
            query="neural networks learning",
            query_embedding=sample_embeddings[0],
            top_k=3,
        )
        for result in results:
            assert result["rrf_score"] > 0

    def test_rrf_scores_ordered(self, vector_store, sample_chunks, sample_embeddings):
        """Test results are ordered by decreasing RRF score"""
        vector_store.add_documents(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="order_doc",
        )
        retriever = HybridRetriever(vector_store)

        results = retriever.retrieve(
            query="test",
            query_embedding=sample_embeddings[0],
            top_k=5,
        )
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

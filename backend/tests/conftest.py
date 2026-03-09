"""Shared test fixtures and configuration"""

import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.vector_store import VectorStore


@pytest.fixture
def doc_processor():
    """Create a DocumentProcessor with default settings"""
    return DocumentProcessor(chunk_size=512, chunk_overlap=50)


@pytest.fixture
def small_doc_processor():
    """Create a DocumentProcessor with small chunks for testing"""
    return DocumentProcessor(chunk_size=40, chunk_overlap=8)


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def vector_store(temp_chroma_dir):
    """Create a VectorStore with temporary storage"""
    return VectorStore(persist_dir=temp_chroma_dir)


@pytest.fixture
def sample_text_content():
    """Sample text file content"""
    return b"This is a sample document about artificial intelligence. " \
           b"Machine learning is a subset of AI. " \
           b"Deep learning is a subset of machine learning. " \
           b"Neural networks power deep learning systems. " \
           b"Natural language processing is another AI field."


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing"""
    return [
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning algorithms learn from data patterns.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret images.",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors (1536-dim like text-embedding-3-small)"""
    import random
    random.seed(42)
    return [[random.uniform(-1, 1) for _ in range(1536)] for _ in range(5)]


@pytest.fixture
def sample_metadata():
    """Sample metadata for chunks"""
    return [
        {"source": "test_doc.txt"},
        {"source": "test_doc.txt"},
        {"source": "test_doc.txt"},
        {"source": "test_doc.txt"},
        {"source": "test_doc.txt"},
    ]

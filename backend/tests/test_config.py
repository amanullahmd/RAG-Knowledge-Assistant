"""Tests for configuration"""

import pytest
from backend.app.core.config import Settings, settings


class TestSettings:
    """Test application settings"""

    def test_settings_instance(self):
        """Test settings singleton is created"""
        assert settings is not None

    def test_default_values(self):
        """Test default setting values"""
        s = Settings()
        assert s.backend_host == "0.0.0.0"
        assert s.backend_port == 8000
        assert s.frontend_port == 8501
        assert s.chunk_size == 512
        assert s.chunk_overlap == 50
        assert s.top_k_retrieval == 5
        assert s.log_level == "INFO"

    def test_default_models(self):
        """Test default model names"""
        s = Settings()
        assert s.embedding_model == "text-embedding-3-small"
        assert s.llm_model == "gpt-4o-mini"

    def test_cors_default(self):
        """Test CORS default"""
        s = Settings()
        assert s.cors_origins == "*"

    def test_chroma_dir_default(self):
        """Test ChromaDB directory default"""
        s = Settings()
        assert s.chroma_persist_dir == "./chroma_data"

    def test_chunk_overlap_less_than_size(self):
        """Test that chunk overlap is less than chunk size"""
        assert settings.chunk_overlap < settings.chunk_size

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: str
    
    # Backend Configuration
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    
    # Frontend Configuration
    frontend_host: str = "0.0.0.0"
    frontend_port: int = 8501
    
    # ChromaDB Configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    top_k_retrieval: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

"""Application configuration from environment variables"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # OpenAI Configuration
    openai_api_key: str = ""

    # Backend Configuration
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    port: int = 0  # Railway sets PORT; 0 means "use backend_port"

    # Frontend Configuration
    frontend_host: str = "0.0.0.0"
    frontend_port: int = 8501

    # ChromaDB Configuration
    chroma_persist_dir: str = "./chroma_data"

    # Logging
    log_level: str = "INFO"

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    top_k_retrieval: int = 5

    # CORS
    cors_origins: str = "*"

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


settings = Settings()

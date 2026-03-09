"""FastAPI application initialization"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    logger.info("Starting up RAG Knowledge Assistant API")
    # Import dependencies to initialize services at startup
    from backend.app.dependencies import vector_store  # noqa: F401

    logger.info(
        f"Vector store initialized with {vector_store.count()} existing chunks"
    )
    yield
    logger.info("Shutting down RAG Knowledge Assistant API")


app = FastAPI(
    title="RAG Knowledge Assistant",
    version="1.0.0",
    description="AI-powered RAG system for company knowledge with hybrid search",
    lifespan=lifespan,
)

# CORS middleware - configurable origins
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "RAG Knowledge Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check with service status"""
    try:
        from backend.app.dependencies import vector_store

        chunk_count = vector_store.count()
        return {
            "status": "healthy",
            "vector_store_chunks": chunk_count,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# Import routers AFTER app creation to avoid circular imports
from backend.app.api.v1.endpoints import documents, chat  # noqa: E402

app.include_router(documents.router)
app.include_router(chat.router)

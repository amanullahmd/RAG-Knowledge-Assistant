#!/usr/bin/env python3
"""Entry point for running the RAG Knowledge Assistant backend."""

import uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )

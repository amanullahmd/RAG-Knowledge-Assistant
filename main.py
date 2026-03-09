#!/usr/bin/env python3
"""Entry point for running the RAG Knowledge Assistant backend."""

import os
import uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    # Railway sets PORT env var; fall back to settings for local dev
    port = int(os.getenv("PORT", settings.backend_port))
    uvicorn.run(
        "backend.app.main:app",
        host=settings.backend_host,
        port=port,
        reload=False,
    )

"""Embedding service using OpenAI"""

import logging
from typing import List
from openai import OpenAI

from backend.app.core.config import settings
from backend.app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using OpenAI API"""

    def __init__(self, api_key: str = None, model: str = None):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []

        try:
            cleaned = [text.replace("\n", " ").strip()[:8191] for text in texts]
            cleaned = [t if t else " " for t in cleaned]

            response = self.client.embeddings.create(
                input=cleaned,
                model=self.model,
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

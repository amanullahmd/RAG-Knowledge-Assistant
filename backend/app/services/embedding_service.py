"""Embedding service using OpenAI"""

import logging
from typing import List
from openai import OpenAI

from backend.app.core.config import settings
from backend.app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Generate embeddings using OpenAI API"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Clean texts
            texts = [text.replace("\n", " ")[:8191] for text in texts]
            
            # Call OpenAI API
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

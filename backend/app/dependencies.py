"""Shared service singletons - all endpoints use the same instances."""

from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store import VectorStore
from backend.app.services.retriever import HybridRetriever
from backend.app.services.llm_service import LLMService
from backend.app.services.chat_service import ChatService

# Shared singleton instances
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()
retriever = HybridRetriever(vector_store)
llm_service = LLMService()
chat_service = ChatService(embedding_service, retriever, llm_service)

# In-memory document metadata store
documents_db: dict = {}

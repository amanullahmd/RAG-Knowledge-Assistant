"""Pydantic models and schemas"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Document Models
class DocumentBase(BaseModel):
    """Base document model"""
    filename: str
    file_type: str
    size_bytes: int

class DocumentCreate(DocumentBase):
    """Document creation schema"""
    content: str

class DocumentResponse(DocumentBase):
    """Document response schema"""
    doc_id: str
    uploaded_at: datetime
    chunks_count: int

    class Config:
        from_attributes = True

# Chunk Models
class ChunkResponse(BaseModel):
    """Chunk response schema"""
    chunk_id: str
    doc_id: str
    content: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

# Chat Models
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatQueryRequest(BaseModel):
    """Chat query request"""
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5

class Citation(BaseModel):
    """Citation model for sources"""
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    content_snippet: str

class ChatQueryResponse(BaseModel):
    """Chat query response"""
    answer: str
    citations: List[Citation]
    session_id: str
    timestamp: datetime

class ChatHistoryResponse(BaseModel):
    """Chat history response"""
    session_id: str
    messages: List[ChatMessage]

# Evaluation Models
class EvaluationMetrics(BaseModel):
    """RAGAS evaluation metrics"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

class EvaluationRequest(BaseModel):
    """Evaluation request"""
    test_dataset: Optional[List[dict]] = None

class EvaluationResponse(BaseModel):
    """Evaluation response"""
    metrics: EvaluationMetrics
    timestamp: datetime

"""Document management API endpoints"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List
from datetime import datetime
import uuid

from backend.app.models.schemas import DocumentResponse, DocumentCreate
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store import VectorStore
from backend.app.services.chat_service import ChatService
from backend.app.core.exceptions import DocumentProcessingError

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# Service instances
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()

# Store document metadata (in-memory for now)
documents_db = {}

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    try:
        doc_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Handle None filename
        filename_str = file.filename or "unknown"
        
        # Process document
        text, chunks, metadata = doc_processor.process_file(filename_str, content)
        
        # Generate embeddings
        embeddings = embedding_service.embed_texts(chunks)
        
        # Add to vector store
        chunk_ids = vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            doc_id=doc_id,
            metadata=metadata
        )
        
        # Store document metadata
        documents_db[doc_id] = {
            "filename": file.filename,
            "doc_id": doc_id,
            "size_bytes": len(content),
            "file_type": file.content_type or "unknown",
            "uploaded_at": datetime.now(),
            "chunks_count": len(chunks)
        }
        
        logger.info(f"Uploaded document: {doc_id} ({file.filename})")
        
        return DocumentResponse(
            doc_id=doc_id,
            filename=filename_str,
            file_type=file.content_type or "unknown",
            size_bytes=len(content),
            uploaded_at=datetime.now(),
            chunks_count=len(chunks)
        )
        
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Upload failed")

@router.get("", response_model=List[DocumentResponse])
async def list_documents():
    """List all uploaded documents"""
    return [DocumentResponse(**doc) for doc in documents_db.values()]

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get a specific document"""
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    
    return DocumentResponse(**documents_db[doc_id])

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks"""
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    
    try:
        # Delete from vector store
        vector_store.delete_document(doc_id)
        
        # Delete metadata
        del documents_db[doc_id]
        
        logger.info(f"Deleted document: {doc_id}")
        
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Delete failed")

"""Document management API endpoints"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from datetime import datetime
import uuid

from backend.app.models.schemas import DocumentResponse
from backend.app.core.exceptions import DocumentProcessingError
from backend.app.dependencies import (
    doc_processor,
    embedding_service,
    vector_store,
    retriever,
    documents_db,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    try:
        doc_id = str(uuid.uuid4())
        content = await file.read()
        filename = file.filename or "unknown"

        # Process document
        text, chunks, metadata = doc_processor.process_file(filename, content)

        # Generate embeddings
        embeddings = embedding_service.embed_texts(chunks)

        # Add to vector store
        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            doc_id=doc_id,
            metadata=metadata,
        )

        # Rebuild BM25 index with new documents
        retriever.rebuild_index()

        # Store document metadata
        doc_meta = {
            "filename": filename,
            "doc_id": doc_id,
            "size_bytes": len(content),
            "file_type": file.content_type or "unknown",
            "uploaded_at": datetime.now(),
            "chunks_count": len(chunks),
        }
        documents_db[doc_id] = doc_meta

        logger.info(f"Uploaded document: {doc_id} ({filename}, {len(chunks)} chunks)")

        return DocumentResponse(**doc_meta)

    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {e.detail}")
        raise HTTPException(status_code=422, detail=e.detail)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


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
        vector_store.delete_document(doc_id)
        del documents_db[doc_id]
        retriever.rebuild_index()

        logger.info(f"Deleted document: {doc_id}")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

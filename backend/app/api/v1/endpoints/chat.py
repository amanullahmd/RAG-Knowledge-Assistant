"""Chat API endpoints"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime

from backend.app.models.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    ChatHistoryResponse,
    ChatMessage,
    Citation,
)
from backend.app.dependencies import chat_service

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/query", response_model=ChatQueryResponse)
async def query(request: ChatQueryRequest):
    """Send a chat query and get an answer with citations"""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        answer, citations, session_id = chat_service.query(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k or 5,
        )

        return ChatQueryResponse(
            answer=answer,
            citations=[
                Citation(
                    source=c["source"],
                    page=c.get("page"),
                    section=c.get("section"),
                    content_snippet=c.get("content_snippet", ""),
                )
                for c in citations
            ],
            session_id=session_id,
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


@router.post("/stream")
async def stream_query(request: ChatQueryRequest):
    """Stream a chat response"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def generate():
        try:
            for chunk in chat_service.query_stream(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k or 5,
            ):
                yield chunk.encode("utf-8")
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"Error: {e}".encode("utf-8")

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = chat_service.get_history(session_id)

        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")

        return ChatHistoryResponse(
            session_id=session_id,
            messages=[
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(
                        msg.get("timestamp", datetime.now().isoformat())
                    ),
                )
                for msg in messages
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get history failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        if chat_service.clear_session(session_id):
            return {"message": "Session deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

"""Chat service for conversation management"""

import json
import logging
import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from backend.app.services.llm_service import LLMService
from backend.app.services.retriever import HybridRetriever
from backend.app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ChatService:
    """Manages chat sessions and RAG conversations"""

    # Quick regex for casual greetings / small-talk that don't need RAG
    _CASUAL_RE = re.compile(
        r"^\s*("
        r"h(i|ello|ey|owdy|ola)"
        r"|good\s*(morning|afternoon|evening|night|day)"
        r"|what'?s\s*up|sup|yo|greetings"
        r"|how\s*are\s*you(\s*doing)?"
        r"|how'?s\s*it\s*going"
        r"|thank(s|\s*you)"
        r"|bye|goodbye|see\s*ya|cheers|later"
        r"|nice\s*to\s*meet\s*you"
        r"|help|help\s*me"
        r"|who\s*are\s*you|what\s*can\s*you\s*do"
        r"|what\s*are\s*you"
        r")\s*[!?.,]*\s*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever: HybridRetriever,
        llm_service: LLMService,
    ):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.llm_service = llm_service
        self.sessions: Dict[str, Dict] = {}

    def _is_casual(self, query: str) -> bool:
        """Return True for greetings / small-talk that don't need document retrieval."""
        return bool(self._CASUAL_RE.match(query.strip()))

    def create_session(self) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": [],
        }
        logger.info(f"Created session: {session_id}")
        return session_id

    def _ensure_session(self, session_id: Optional[str]) -> Tuple[str, Dict]:
        """Ensure session exists, create if needed. Returns (session_id, session)."""
        if not session_id or session_id not in self.sessions:
            session_id = self.create_session()
        return session_id, self.sessions[session_id]

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Tuple[str, List[Dict], str]:
        """Process a chat query through the RAG pipeline."""
        try:
            session_id, session = self._ensure_session(session_id)

            session["messages"].append(
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Skip expensive retrieval for casual greetings
            if self._is_casual(query):
                context_chunks = []
            else:
                query_embedding = self.embedding_service.embed_text(query)
                context_chunks = self.retriever.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )

            answer, citations = self.llm_service.generate_answer(
                query=query,
                context_chunks=context_chunks,
                conversation_history=session["messages"][:-1],
            )

            session["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                    "citations": citations,
                }
            )

            if len(session["messages"]) > 20:
                session["messages"] = session["messages"][-20:]

            return answer, citations, session_id

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

    def query_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ):
        """Stream a chat query response. Yields chunks of answer text."""
        try:
            session_id, session = self._ensure_session(session_id)

            session["messages"].append(
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Skip expensive retrieval for casual greetings
            if self._is_casual(query):
                context_chunks = []
            else:
                query_embedding = self.embedding_service.embed_text(query)
                context_chunks = self.retriever.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )

            full_response = ""
            for chunk in self.llm_service.generate_answer_stream(
                query=query,
                context_chunks=context_chunks,
                conversation_history=session["messages"][:-1],
            ):
                full_response += chunk
                yield chunk

            citations = self.llm_service.extract_citations(context_chunks)

            # Yield a special marker with citations JSON at the end
            yield f"\n\n__CITATIONS__{json.dumps(citations)}"

            session["messages"].append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().isoformat(),
                    "citations": citations,
                }
            )

            if len(session["messages"]) > 20:
                session["messages"] = session["messages"][-20:]

        except Exception as e:
            logger.error(f"Stream query failed: {e}")
            raise

    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["messages"]

    def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False

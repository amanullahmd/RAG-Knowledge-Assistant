"""Chat service for conversation management"""

import logging
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .llm_service import LLMService
from .retriever import HybridRetriever
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class ChatService:
    """Manages chat sessions and RAG conversations"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever: HybridRetriever,
        llm_service: LLMService
    ):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.llm_service = llm_service
        self.sessions: Dict[str, Dict] = {}  # In-memory session storage
    
    def create_session(self) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": []
        }
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> Tuple[str, List[Dict], str]:
        """
        Process a chat query with RAG
        
        Args:
            query: User query
            session_id: Session ID (creates new if not provided)
            top_k: Number of context chunks
            
        Returns:
            Tuple of (answer, citations, session_id)
        """
        try:
            # Create session if needed
            if not session_id or session_id not in self.sessions:
                session_id = self.create_session()
            
            # Get or create session
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "created_at": datetime.now(),
                    "messages": []
                }
            
            session = self.sessions[session_id]
            
            # Add user message to history
            session["messages"].append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Embed query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Retrieve context
            context_chunks = self.retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Generate answer
            answer, citations = self.llm_service.generate_answer(
                query=query,
                context_chunks=context_chunks,
                conversation_history=session["messages"][:-1]  # Exclude current user message
            )
            
            # Add assistant message to history
            session["messages"].append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat(),
                "citations": [c for c in citations]
            })
            
            # Keep only last 20 messages per session
            if len(session["messages"]) > 20:
                session["messages"] = session["messages"][-20:]
            
            return answer, citations, session_id
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise
    
    def query_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Stream a chat query response
        
        Yields:
            Chunks of the answer text
        """
        try:
            # Create session if needed
            if not session_id or session_id not in self.sessions:
                session_id = self.create_session()
            
            session = self.sessions[session_id]
            
            # Add user message
            session["messages"].append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Embed and retrieve
            query_embedding = self.embedding_service.embed_text(query)
            context_chunks = self.retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Stream answer
            full_response = ""
            for chunk in self.llm_service.generate_answer_stream(
                query=query,
                context_chunks=context_chunks,
                conversation_history=session["messages"][:-1]
            ):
                full_response += chunk
                yield chunk
            
            # Extract citations and add to history
            citations = self.llm_service._extract_citations(full_response, context_chunks)
            session["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat(),
                "citations": citations
            })
            
            if len(session["messages"]) > 20:
                session["messages"] = session["messages"][-20:]
            
        except Exception as e:
            logger.error(f"Stream query failed: {str(e)}")
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

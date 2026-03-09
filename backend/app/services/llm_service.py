"""LLM service for RAG generation with citations"""

import logging
import json
from typing import List, Dict, Tuple
from openai import OpenAI

from backend.app.core import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Generate answers using OpenAI with source citations"""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. Answer ONLY based on the provided context documents
2. Always cite your sources in the format: [Source: filename, Page: X] or [Source: filename]
3. If information is not in the documents, explicitly say "I don't have this information in the provided documents"
4. Be concise but informative
5. Group multiple sources if needed: [Sources: file1.pdf, file2.docx]"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Generate an answer based on context chunks with citations
        
        Args:
            query: User query
            context_chunks: List of retrieved chunks with metadata
            conversation_history: Previous conversation messages
            
        Returns:
            Tuple of (answer, citations)
        """
        try:
            # Prepare context
            context_text = self._prepare_context(context_chunks)
            
            # Build messages
            messages = []
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-4:]:  # Keep last 4 turns for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add current query with context
            user_message = f"""Based on the following documents, answer this question:

QUESTION: {query}

DOCUMENTS:
{context_text}

Please provide a detailed answer citing the specific sources."""
            
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT}] + messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Extract citations
            citations = self._extract_citations(answer, context_chunks)
            
            return answer, citations
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise
    
    def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ):
        """
        Generate answer with streaming
        
        Yields:
            Chunks of the answer text
        """
        try:
            context_text = self._prepare_context(context_chunks)
            
            messages = []
            if conversation_history:
                for msg in conversation_history[-4:]:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            user_message = f"""Based on the following documents, answer this question:

QUESTION: {query}

DOCUMENTS:
{context_text}

Please provide a detailed answer citing the specific sources."""
            
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Stream response
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT}] + messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Format context chunks for the prompt"""
        context = ""
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page", "")
            
            context += f"\n[{i}] Source: {source}"
            if page:
                context += f" (Page {page})"
            context += f"\n{chunk['content']}\n"
        
        return context.strip()
    
    def _extract_citations(self, answer: str, context_chunks: List[Dict]) -> List[Dict]:
        """Extract and format citations from answer and context"""
        citations = []
        sources_seen = set()
        
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page")
            
            # Check if this source is referenced in answer or use all sources
            source_key = f"{source}_{page}"
            if source_key not in sources_seen:
                citation = {
                    "source": source,
                    "page": page,
                    "section": metadata.get("section"),
                    "content_snippet": chunk["content"][:200]  # First 200 chars
                }
                citations.append(citation)
                sources_seen.add(source_key)
        
        return citations

"""LLM service for RAG generation with citations"""

import logging
from typing import List, Dict, Tuple
from openai import OpenAI

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Generate answers using OpenAI with source citations"""

    SYSTEM_PROMPT = """You are Knowledge AI — a friendly, professional assistant that helps users explore their uploaded documents through natural conversation.

GUIDELINES:
1. **Casual messages** (greetings like "hi", "hello", "how are you", small talk, thanks):
   Respond warmly and naturally. Introduce yourself briefly and let the user know you're ready to help with their documents.

2. **Document-based questions**:
   - Answer using the provided context documents.
   - Cite every claim: [Source: filename, Page: X] or [Source: filename].
   - Group multiple sources: [Sources: file1.pdf, file2.docx].

3. **Information not found**: If the documents don't contain the answer, say so clearly but remain helpful — suggest rephrasing or uploading additional documents.

4. Be concise, accurate, and conversational."""

    def __init__(self, api_key: str = None, model: str = None):
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.llm_model

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None,
    ) -> Tuple[str, List[Dict]]:
        """Generate an answer based on context chunks with citations."""
        try:
            context_text = self._prepare_context(context_chunks)

            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

            if conversation_history:
                for msg in conversation_history[-4:]:
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            user_message = self._build_user_message(query, context_text)
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content or ""
            citations = self.extract_citations(context_chunks)

            return answer, citations

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None,
    ):
        """Generate answer with streaming. Yields chunks of answer text."""
        try:
            context_text = self._prepare_context(context_chunks)

            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

            if conversation_history:
                for msg in conversation_history[-4:]:
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            user_message = self._build_user_message(query, context_text)
            messages.append({"role": "user", "content": user_message})

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def _build_user_message(self, query: str, context_text: str) -> str:
        """Build an adaptive user message depending on available context."""
        if context_text == "(No relevant documents found)":
            # No docs — let the LLM respond naturally (greetings, general chat)
            return query

        return (
            f"QUESTION: {query}\n\n"
            f"CONTEXT DOCUMENTS:\n{context_text}\n\n"
            f"Respond to the user naturally. If the question relates to the documents, "
            f"answer using the context above and cite sources. "
            f"If it is a greeting or casual message, respond warmly."
        )

    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Format context chunks for the prompt"""
        if not chunks:
            return "(No relevant documents found)"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page", "")

            header = f"[{i}] Source: {source}"
            if page:
                header += f" (Page {page})"
            parts.append(f"{header}\n{chunk['content']}")

        return "\n\n".join(parts)

    def extract_citations(self, context_chunks: List[Dict]) -> List[Dict]:
        """Extract and format citations from context chunks"""
        citations = []
        sources_seen = set()

        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page")

            source_key = f"{source}_{page}"
            if source_key not in sources_seen:
                citations.append(
                    {
                        "source": source,
                        "page": page,
                        "section": metadata.get("section"),
                        "content_snippet": chunk["content"][:200],
                    }
                )
                sources_seen.add(source_key)

        return citations

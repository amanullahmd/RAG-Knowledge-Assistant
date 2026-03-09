"""Vector store service using ChromaDB"""

import logging
from typing import List, Dict, Optional
import chromadb

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for RAG"""

    def __init__(self, persist_dir: str = None):
        """Initialize ChromaDB with persistent storage"""
        persist_dir = persist_dir or settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        doc_id: str,
        metadata: List[Dict] = None,
    ) -> List[str]:
        """Add document chunks to vector store"""
        if not chunks:
            return []

        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        if metadata is None:
            metadata = [{"doc_id": doc_id} for _ in chunks]
        else:
            for m in metadata:
                m["doc_id"] = doc_id

        try:
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
            )
            logger.info(f"Added {len(chunks)} chunks for document {doc_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(
        self, embedding: List[float], top_k: int = 5, doc_id: str = None
    ) -> List[Dict]:
        """Search for similar chunks"""
        try:
            where = {"doc_id": doc_id} if doc_id else None

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, self.collection.count() or 1),
                where=where,
            )

            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append(
                        {
                            "chunk_id": chunk_id,
                            "content": results["documents"][0][i],
                            "distance": (
                                results["distances"][0][i]
                                if results["distances"]
                                else 0
                            ),
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else {}
                            ),
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def get_all_documents(self) -> Dict:
        """Get all documents and their IDs from the collection"""
        try:
            return self.collection.get()
        except Exception as e:
            logger.error(f"Get all documents failed: {e}")
            raise

    def get_by_doc_id(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            results = self.collection.get(where={"doc_id": doc_id})

            formatted_results = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    formatted_results.append(
                        {
                            "chunk_id": chunk_id,
                            "content": results["documents"][i],
                            "metadata": (
                                results["metadatas"][i]
                                if results["metadatas"]
                                else {}
                            ),
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Get by doc_id failed: {e}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            results = self.collection.get(where={"doc_id": doc_id})

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {doc_id}"
                )

            return True
        except Exception as e:
            logger.error(f"Delete document failed: {e}")
            raise

    def clear(self):
        """Clear all documents"""
        try:
            self.client.delete_collection(name="documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            raise

    def count(self) -> int:
        """Return the number of documents in the collection"""
        return self.collection.count()

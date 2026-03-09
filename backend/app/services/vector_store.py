"""Vector store service using ChromaDB"""

import logging
import uuid
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.core import settings as app_settings

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB vector store for RAG"""
    
    def __init__(self, persist_dir: str = "./chroma_data"):
        """Initialize ChromaDB"""
        # Use persistent storage
        self.client = chromadb.Client(
            ChromaSettings(
                is_persistent=True,
                persist_directory=persist_dir,
                anonymized_telemetry=False
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        doc_id: str,
        metadata: List[Dict] = None
    ) -> List[str]:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            doc_id: Document ID
            metadata: List of metadata dicts
            
        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []
        
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        # Prepare metadata
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
                metadatas=metadata
            )
            logger.info(f"Added {len(chunks)} chunks for document {doc_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def search(self, embedding: List[float], top_k: int = 5, doc_id: str = None) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            doc_id: Optional filter by document ID
            
        Returns:
            List of search results with content and metadata
        """
        try:
            where = {"doc_id": doc_id} if doc_id else None
            
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "chunk_id": chunk_id,
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def get_by_doc_id(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            formatted_results = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    formatted_results.append({
                        "chunk_id": chunk_id,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Get by doc_id failed: {str(e)}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # Get all chunks for this doc
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
            
            return True
        except Exception as e:
            logger.error(f"Delete document failed: {str(e)}")
            raise
    
    def clear(self):
        """Clear all documents"""
        try:
            # Delete collection and recreate
            self.client.delete_collection(name="documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Clear failed: {str(e)}")
            raise

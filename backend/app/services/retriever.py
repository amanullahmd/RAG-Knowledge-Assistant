"""Hybrid retriever combining vector search and BM25"""

import logging
from typing import List, Dict
from rank_bm25 import BM25Okapi

from backend.app.core.exceptions import RetrievalError
from backend.app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines vector similarity and BM25 keyword search with Reciprocal Rank Fusion"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25_index = None
        self.documents = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in vector store"""
        try:
            # Get all documents
            results = self.vector_store.collection.get()
            
            if results["documents"]:
                # Tokenize documents
                self.documents = results["documents"]
                tokenized_docs = [doc.split() for doc in self.documents]
                self.bm25_index = BM25Okapi(tokenized_docs)
                logger.info(f"Built BM25 index with {len(self.documents)} documents")
            else:
                self.bm25_index = None
                self.documents = []
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {str(e)}")
    
    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve using hybrid search (vector + BM25 + RRF)
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of retrieved documents
        """
        try:
            results = {}
            
            # Vector search
            vector_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
            for i, result in enumerate(vector_results):
                chunk_id = result["chunk_id"]
                if chunk_id not in results:
                    results[chunk_id] = {"vector_rank": i + 1, "bm25_rank": float('inf')}
                else:
                    results[chunk_id]["vector_rank"] = i + 1
            
            # BM25 search
            if self.bm25_index:
                query_tokens = query.split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                
                # Get top-k BM25 results
                top_bm25_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:top_k * 2]
                
                for rank, idx in enumerate(top_bm25_indices):
                    if idx < len(self.documents):
                        chunk_id = None
                        # Find chunk_id from document
                        for result in vector_results:
                            if result["content"] == self.documents[idx]:
                                chunk_id = result["chunk_id"]
                                break
                        
                        # Also check via vector store
                        if not chunk_id:
                            vec_results = self.vector_store.collection.get()
                            for i, doc in enumerate(vec_results["documents"]):
                                if doc == self.documents[idx] and i < len(vec_results["ids"]):
                                    chunk_id = vec_results["ids"][i]
                                    break
                        
                        if chunk_id:
                            if chunk_id not in results:
                                results[chunk_id] = {"vector_rank": float('inf'), "bm25_rank": rank + 1}
                            else:
                                results[chunk_id]["bm25_rank"] = rank + 1
            
            # Calculate RRF scores and rank
            rrf_scored = []
            for chunk_id, ranks in results.items():
                vector_rank = ranks.get("vector_rank", float('inf'))
                bm25_rank = ranks.get("bm25_rank", float('inf'))
                
                # RRF formula: 1 / (k + rank) for each ranker
                k = 60
                rrf_score = 0
                if vector_rank != float('inf'):
                    rrf_score += 1 / (k + vector_rank)
                if bm25_rank != float('inf'):
                    rrf_score += 1 / (k + bm25_rank)
                
                rrf_scored.append((chunk_id, rrf_score))
            
            # Sort by RRF score
            rrf_scored.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k and return full results
            final_results = []
            for chunk_id, score in rrf_scored[:top_k]:
                # Find the full result
                for result in vector_results:
                    if result["chunk_id"] == chunk_id:
                        result["rrf_score"] = score
                        final_results.append(result)
                        break
                
                # If not found in vector results, fetch from store
                if not any(r["chunk_id"] == chunk_id for r in final_results):
                    vec_data = self.vector_store.collection.get(ids=[chunk_id])
                    if vec_data["ids"]:
                        final_results.append({
                            "chunk_id": chunk_id,
                            "content": vec_data["documents"][0],
                            "metadata": vec_data["metadatas"][0] if vec_data["metadatas"] else {},
                            "rrf_score": score
                        })
            
            return final_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RetrievalError(str(e))

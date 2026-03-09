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
        self.doc_texts: List[str] = []
        self.doc_ids: List[str] = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from all documents in vector store"""
        try:
            results = self.vector_store.get_all_documents()

            if results["documents"]:
                self.doc_texts = results["documents"]
                self.doc_ids = results["ids"]
                tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
                self.bm25_index = BM25Okapi(tokenized_docs)
                logger.info(f"Built BM25 index with {len(self.doc_texts)} chunks")
            else:
                self.bm25_index = None
                self.doc_texts = []
                self.doc_ids = []
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
            self.doc_texts = []
            self.doc_ids = []

    def rebuild_index(self):
        """Rebuild BM25 index (call after adding new documents)"""
        self._build_bm25_index()

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve using hybrid search (vector + BM25 + RRF).

        Returns:
            List of retrieved documents with content, metadata, and rrf_score
        """
        try:
            # Rebuild BM25 index to include any newly added documents
            self._build_bm25_index()

            results: Dict[str, Dict] = {}
            vector_results_map: Dict[str, Dict] = {}

            # Vector search
            vector_results = self.vector_store.search(
                query_embedding, top_k=min(top_k * 4, 20)
            )
            for i, result in enumerate(vector_results):
                chunk_id = result["chunk_id"]
                vector_results_map[chunk_id] = result
                if chunk_id not in results:
                    results[chunk_id] = {
                        "vector_rank": i + 1,
                        "bm25_rank": float("inf"),
                    }
                else:
                    results[chunk_id]["vector_rank"] = i + 1

            # BM25 search
            if self.bm25_index and self.doc_texts:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)

                top_bm25_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )[: min(top_k * 4, 20)]

                for rank, idx in enumerate(top_bm25_indices):
                    if idx < len(self.doc_ids) and bm25_scores[idx] > 0:
                        chunk_id = self.doc_ids[idx]
                        if chunk_id not in results:
                            results[chunk_id] = {
                                "vector_rank": float("inf"),
                                "bm25_rank": rank + 1,
                            }
                        else:
                            results[chunk_id]["bm25_rank"] = rank + 1

            # Calculate RRF scores
            k = 60
            rrf_scored = []
            for chunk_id, ranks in results.items():
                rrf_score = 0.0
                vector_rank = ranks.get("vector_rank", float("inf"))
                bm25_rank = ranks.get("bm25_rank", float("inf"))

                if vector_rank != float("inf"):
                    rrf_score += 1.0 / (k + vector_rank)
                if bm25_rank != float("inf"):
                    rrf_score += 1.0 / (k + bm25_rank)

                rrf_scored.append((chunk_id, rrf_score))

            rrf_scored.sort(key=lambda x: x[1], reverse=True)

            # Get top-k results
            final_results = []
            for chunk_id, score in rrf_scored[:top_k]:
                if chunk_id in vector_results_map:
                    result = dict(vector_results_map[chunk_id])
                    result["rrf_score"] = score
                    final_results.append(result)
                else:
                    # Fetch from store for BM25-only results
                    try:
                        vec_data = self.vector_store.collection.get(ids=[chunk_id])
                        if vec_data["ids"]:
                            final_results.append(
                                {
                                    "chunk_id": chunk_id,
                                    "content": vec_data["documents"][0],
                                    "metadata": (
                                        vec_data["metadatas"][0]
                                        if vec_data["metadatas"]
                                        else {}
                                    ),
                                    "rrf_score": score,
                                }
                            )
                    except Exception:
                        logger.warning(f"Could not fetch chunk {chunk_id}")

            return final_results

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(str(e))

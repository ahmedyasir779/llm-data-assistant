from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class HybridSearchEngine:
    """
    Combines multiple search strategies for optimal results
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid search engine
        
        Args:
            alpha: Weight for semantic vs keyword (0=keyword only, 1=semantic only)
        """
        self.alpha = alpha  # Weight for combining scores
        self.bm25_index = None
        self.documents = []
        self.metadatas = []
        
        print(f"🔍 Hybrid search initialized (alpha={alpha})")
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Index documents for keyword search
        
        Args:
            documents: List of document strings
            metadatas: List of metadata dicts
        """
        self.documents = documents
        self.metadatas = metadatas
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        print(f"✅ Indexed {len(documents)} documents for hybrid search")
    
    def search(
        self,
        query: str,
        semantic_results: Dict[str, Any],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining BM25 and semantic search
        
        Args:
            query: Search query
            semantic_results: Results from semantic search (ChromaDB)
            n_results: Number of results to return
            
        Returns:
            Combined search results
        """
        if not self.bm25_index:
            # No keyword index, return semantic only
            return semantic_results
        
        # Get BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Get semantic scores (convert distances to similarities)
        semantic_scores = np.array([
            1.0 - d for d in semantic_results.get("distances", [])
        ])
        
        # Combine scores
        combined_scores = (
            self.alpha * semantic_scores[:len(bm25_scores)] +
            (1 - self.alpha) * bm25_scores[:len(semantic_scores)]
        )
        
        # Get top N indices
        top_indices = np.argsort(combined_scores)[-n_results:][::-1]
        
        # Build results
        return {
            "documents": [semantic_results["documents"][i] for i in top_indices],
            "metadatas": [semantic_results["metadatas"][i] for i in top_indices],
            "scores": combined_scores[top_indices].tolist(),
            "bm25_scores": bm25_scores[top_indices].tolist(),
            "semantic_scores": semantic_scores[top_indices].tolist()
        }
    
    def keyword_search_only(
        self,
        query: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Perform keyword search only (BM25)
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Search results
        """
        if not self.bm25_index:
            return {"documents": [], "metadatas": [], "scores": []}
        
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top N
        top_indices = np.argsort(scores)[-n_results:][::-1]
        
        return {
            "documents": [self.documents[i] for i in top_indices],
            "metadatas": [self.metadatas[i] for i in top_indices],
            "scores": scores[top_indices].tolist()
        }
    
    def set_alpha(self, alpha: float) -> None:
        """
        Update alpha (semantic/keyword weight)
        
        Args:
            alpha: New alpha value (0-1)
        """
        self.alpha = max(0.0, min(1.0, alpha))
        print(f"✅ Alpha updated to {self.alpha}")


class ResultReranker:
    """
    Re-ranks search results based on relevance and quality
    """
    
    def __init__(self):
        """Initialize reranker"""
        print("📊 Result reranker initialized")
    
    def rerank(
        self,
        query: str,
        results: Dict[str, Any],
        boost_metadata: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Re-rank search results
        
        Args:
            query: Original query
            results: Search results to re-rank
            boost_metadata: Metadata fields to boost with weights
            
        Returns:
            Re-ranked results
        """
        if not results.get("documents"):
            return results
        
        # Calculate re-ranking scores
        rerank_scores = []
        
        for idx, (doc, meta, score) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results.get("scores", [1.0] * len(results["documents"]))
        )):
            # Start with original score
            new_score = score
            
            # Boost based on document type
            type_boost = self._get_type_boost(meta.get("type", ""))
            new_score *= type_boost
            
            # Boost based on metadata
            if boost_metadata:
                meta_boost = self._get_metadata_boost(meta, boost_metadata)
                new_score *= meta_boost
            
            # Boost based on query term matches
            term_boost = self._get_term_boost(query, doc)
            new_score *= term_boost
            
            rerank_scores.append(new_score)
        
        # Sort by new scores
        sorted_indices = np.argsort(rerank_scores)[::-1]
        
        return {
            "documents": [results["documents"][i] for i in sorted_indices],
            "metadatas": [results["metadatas"][i] for i in sorted_indices],
            "scores": [rerank_scores[i] for i in sorted_indices],
            "original_scores": [results.get("scores", [1.0] * len(results["documents"]))[i] 
                              for i in sorted_indices]
        }
    
    def _get_type_boost(self, doc_type: str) -> float:
        """Boost score based on document type"""
        boosts = {
            "column_metadata": 1.2,
            "data_chunk": 1.5,
            "statistics": 1.1,
            "categorical_summary": 1.0
        }
        return boosts.get(doc_type, 1.0)
    
    def _get_metadata_boost(
        self,
        metadata: Dict[str, Any],
        boost_fields: Dict[str, float]
    ) -> float:
        """Boost based on metadata fields"""
        boost = 1.0
        
        for field, weight in boost_fields.items():
            if field in metadata:
                boost *= weight
        
        return boost
    
    def _get_term_boost(self, query: str, document: str) -> float:
        """Boost based on exact term matches"""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        # Calculate term overlap
        overlap = len(query_terms & doc_terms)
        total = len(query_terms)
        
        if total == 0:
            return 1.0
        
        # Boost based on overlap percentage
        overlap_ratio = overlap / total
        return 1.0 + (overlap_ratio * 0.5)  # Up to 50% boost


class SearchRouter:
    """
    Routes queries to optimal search strategy
    """
    
    def __init__(
        self,
        hybrid_engine: HybridSearchEngine,
        reranker: ResultReranker
    ):
        """
        Initialize search router
        
        Args:
            hybrid_engine: Hybrid search engine
            reranker: Result reranker
        """
        self.hybrid_engine = hybrid_engine
        self.reranker = reranker
        
        print("🚦 Search router initialized")
    
    def route_and_search(
        self,
        query: str,
        classification: Dict[str, Any],
        semantic_results: Dict[str, Any],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Route query to best search strategy and execute
        
        Args:
            query: User query
            classification: Query classification
            semantic_results: Results from semantic search
            n_results: Number of results
            
        Returns:
            Final search results
        """
        strategy = classification["search_strategy"]
        
        # Execute based on strategy
        if strategy == "semantic":
            results = semantic_results
        
        elif strategy == "keyword":
            results = self.hybrid_engine.keyword_search_only(query, n_results)
        
        elif strategy == "hybrid":
            results = self.hybrid_engine.search(
                query,
                semantic_results,
                n_results
            )
        
        else:  # direct
            results = semantic_results
        
        # Re-rank results
        final_results = self.reranker.rerank(query, results)
        
        # Add metadata about routing
        final_results["search_strategy"] = strategy
        final_results["query_type"] = classification["query_type"].value
        
        return final_results
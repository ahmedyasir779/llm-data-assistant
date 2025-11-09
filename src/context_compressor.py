from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContextCompressor:
    def __init__(self, compression_ratio: float = 0.5):
        """
        Initialize context compressor
        
        Args:
            compression_ratio: Target compression (0.5 = 50% reduction)
        """
        self.compression_ratio = compression_ratio
        self.vectorizer = TfidfVectorizer(max_features=100)
        
        print(f"🗜️ Context compressor initialized")
        print(f"   Compression ratio: {compression_ratio:.0%}")
    
    def compress_context(
        self,
        documents: List[str],
        query: str,
        max_docs: Optional[int] = None
    ) -> List[str]:
        """
        Compress context by selecting most relevant documents
        
        Args:
            documents: List of context documents
            query: User query
            max_docs: Maximum documents to keep
            
        Returns:
            Compressed list of documents
        """
        if not documents:
            return documents
        
        # Calculate target size
        if max_docs is None:
            max_docs = max(1, int(len(documents) * self.compression_ratio))
        
        if len(documents) <= max_docs:
            return documents
        
        # Calculate relevance scores
        scores = self._calculate_relevance_scores(documents, query)
        
        # Get top documents
        top_indices = np.argsort(scores)[-max_docs:][::-1]
        
        return [documents[i] for i in sorted(top_indices)]
    
    def compress_text(
        self,
        text: str,
        max_sentences: int = 10
    ) -> str:
        """
        Compress single text by extractive summarization
        
        Args:
            text: Text to compress
            max_sentences: Maximum sentences to keep
            
        Returns:
            Compressed text
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by length and position
        scores = []
        for idx, sent in enumerate(sentences):
            # Longer sentences tend to have more info
            length_score = len(sent.split()) / 20.0  # Normalize
            # Earlier sentences often more important
            position_score = 1.0 - (idx / len(sentences))
            # Combined score
            score = 0.6 * length_score + 0.4 * position_score
            scores.append(score)
        
        # Get top sentences
        top_indices = np.argsort(scores)[-max_sentences:]
        top_indices = sorted(top_indices)  # Keep original order
        
        compressed = '. '.join([sentences[i] for i in top_indices])
        return compressed + '.'
    
    def remove_duplicates(
        self,
        documents: List[str],
        similarity_threshold: float = 0.9
    ) -> List[str]:
        """
        Remove duplicate or highly similar documents
        
        Args:
            documents: List of documents
            similarity_threshold: Similarity threshold for duplicates
            
        Returns:
            Deduplicated documents
        """
        if len(documents) <= 1:
            return documents
        
        # Calculate similarity matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarities = cosine_similarity(tfidf_matrix)
        except:
            # Fallback if TF-IDF fails
            return documents
        
        # Find duplicates
        keep = [True] * len(documents)
        
        for i in range(len(documents)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(documents)):
                if similarities[i, j] > similarity_threshold:
                    keep[j] = False
        
        return [doc for doc, k in zip(documents, keep) if k]
    
    def _calculate_relevance_scores(
        self,
        documents: List[str],
        query: str
    ) -> np.ndarray:
        """Calculate relevance scores for documents"""
        try:
            # Create TF-IDF matrix
            all_texts = documents + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity to query
            doc_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1]
            
            similarities = cosine_similarity(doc_vectors, query_vector).flatten()
            
            return similarities
            
        except Exception as e:
            # Fallback: score by query term overlap
            query_terms = set(query.lower().split())
            scores = []
            
            for doc in documents:
                doc_terms = set(doc.lower().split())
                overlap = len(query_terms & doc_terms)
                scores.append(overlap / len(query_terms) if query_terms else 0)
            
            return np.array(scores)
    
    def smart_truncate(
        self,
        text: str,
        max_length: int,
        preserve_ends: bool = True
    ) -> str:
        """
        Intelligently truncate text
        
        Args:
            text: Text to truncate
            max_length: Maximum length in characters
            preserve_ends: Whether to preserve beginning and end
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        if preserve_ends:
            # Keep beginning and end
            keep_each = max_length // 2 - 10  # Reserve for ellipsis
            return f"{text[:keep_each]}... [truncated] ...{text[-keep_each:]}"
        else:
            # Keep beginning only
            return f"{text[:max_length-15]}... [truncated]"


class RelevanceFilter:
    """
    Filters context by relevance to query
    """
    
    def __init__(self, min_relevance: float = 0.3):
        """
        Initialize relevance filter
        
        Args:
            min_relevance: Minimum relevance score (0-1)
        """
        self.min_relevance = min_relevance
        self.vectorizer = TfidfVectorizer(max_features=50)
        
        print(f"🎯 Relevance filter initialized")
        print(f"   Min relevance: {min_relevance:.0%}")
    
    def filter_by_relevance(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        query: str,
        scores: Optional[List[float]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Filter documents by relevance to query
        
        Args:
            documents: List of documents
            metadatas: Document metadata
            query: User query
            scores: Optional existing scores
            
        Returns:
            Tuple of (filtered_docs, filtered_metas, filtered_scores)
        """
        if not documents:
            return documents, metadatas, scores or []
        
        # Calculate relevance if not provided
        if scores is None:
            scores = self._calculate_relevance(documents, query)
        
        # Filter by threshold
        filtered = [
            (doc, meta, score)
            for doc, meta, score in zip(documents, metadatas, scores)
            if score >= self.min_relevance
        ]
        
        if not filtered:
            # Return top 1 if all below threshold
            best_idx = np.argmax(scores)
            return (
                [documents[best_idx]],
                [metadatas[best_idx]],
                [scores[best_idx]]
            )
        
        docs, metas, scores = zip(*filtered)
        return list(docs), list(metas), list(scores)
    
    def filter_by_type(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        preferred_types: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Filter documents by metadata type
        
        Args:
            documents: List of documents
            metadatas: Document metadata
            preferred_types: List of preferred document types
            
        Returns:
            Tuple of (filtered_docs, filtered_metas)
        """
        filtered = [
            (doc, meta)
            for doc, meta in zip(documents, metadatas)
            if meta.get("type") in preferred_types
        ]
        
        if not filtered:
            return documents, metadatas
        
        docs, metas = zip(*filtered)
        return list(docs), list(metas)
    
    def _calculate_relevance(
        self,
        documents: List[str],
        query: str
    ) -> List[float]:
        """Calculate relevance scores"""
        try:
            all_texts = documents + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            doc_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1]
            
            similarities = cosine_similarity(doc_vectors, query_vector).flatten()
            
            return similarities.tolist()
            
        except:
            # Fallback
            query_terms = set(query.lower().split())
            scores = []
            
            for doc in documents:
                doc_terms = set(doc.lower().split())
                overlap = len(query_terms & doc_terms)
                score = overlap / len(query_terms) if query_terms else 0.5
                scores.append(score)
            
            return scores
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
import re


class MultiQueryRetriever:
    """
    Generates multiple query variations for comprehensive retrieval
    """
    
    def __init__(self, num_queries: int = 3):
        """
        Initialize multi-query retriever
        
        Args:
            num_queries: Number of query variations to generate
        """
        self.num_queries = num_queries
        
        # Query templates for different perspectives
        self.templates = [
            "{query}",  # Original
            "What information about {query}",  # Explicit
            "Find data related to {query}",  # Search-focused
            "Show me details on {query}",  # Direct
            "Analyze {query}",  # Analysis-focused
        ]
        
        print(f"🔍 Multi-query retriever initialized")
        print(f"   Query variations: {num_queries}")
    
    def generate_queries(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations
        
        Args:
            original_query: Original user query
            
        Returns:
            List of query variations
        """
        queries = [original_query]  # Always include original
        
        # Extract key terms
        key_terms = self._extract_key_terms(original_query)
        
        # Generate variations using templates
        for template in self.templates[1:self.num_queries]:
            variation = template.format(query=original_query.lower())
            queries.append(variation)
        
        # Add keyword-focused queries
        if key_terms:
            keyword_query = " ".join(key_terms)
            if keyword_query not in queries:
                queries.append(keyword_query)
        
        # Add question variations
        question_variations = self._create_question_variations(original_query)
        queries.extend(question_variations[:self.num_queries - len(queries)])
        
        return queries[:self.num_queries]
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important terms from query"""
        stop_words = {
            'what', 'which', 'where', 'when', 'who', 'how', 'why',
            'show', 'tell', 'find', 'get', 'the', 'a', 'an', 'is', 'are'
        }
        
        words = query.lower().split()
        key_terms = [
            w.strip('?.,!') for w in words 
            if w not in stop_words and len(w) > 2
        ]
        
        return key_terms[:5]  # Top 5
    
    def _create_question_variations(self, query: str) -> List[str]:
        """Create question variations"""
        variations = []
        
        # If not a question, make it one
        if not query.strip().endswith('?'):
            # Convert to "What" question
            variations.append(f"What {query.lower()}?")
            # Convert to "How" question
            variations.append(f"How to {query.lower()}?")
        else:
            # Rephrase existing question
            if query.lower().startswith('what'):
                variations.append(query.replace('what', 'which', 1))
            elif query.lower().startswith('how'):
                variations.append(query.replace('how', 'what is the way to', 1))
        
        return variations
    
    def retrieve_and_merge(
        self,
        queries: List[str],
        retriever_func: callable,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve results for all queries and merge
        
        Args:
            queries: List of query variations
            retriever_func: Function to retrieve results (takes query, returns results)
            n_results: Number of final results
            
        Returns:
            Merged results
        """
        all_results = []
        seen_docs = set()
        
        for query in queries:
            try:
                results = retriever_func(query, n_results=n_results * 2)
                
                # Collect unique documents
                for doc, meta, score in zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                    results.get("scores", [1.0] * len(results.get("documents", [])))
                ):
                    # Use first 50 chars as doc identifier
                    doc_id = doc[:50]
                    
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        all_results.append((doc, meta, score, query))
                
            except Exception as e:
                print(f"   ⚠️ Query failed: {query[:30]}... - {e}")
                continue
        
        # Sort by score and take top N
        all_results.sort(key=lambda x: x[2], reverse=True)
        top_results = all_results[:n_results]
        
        if not top_results:
            return {"documents": [], "metadatas": [], "scores": [], "source_queries": []}
        
        docs, metas, scores, source_queries = zip(*top_results)
        
        return {
            "documents": list(docs),
            "metadatas": list(metas),
            "scores": list(scores),
            "source_queries": list(source_queries),
            "num_queries_used": len(queries),
            "unique_results": len(all_results)
        }


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries
    """
    
    def __init__(self):
        """Initialize query decomposer"""
        print("🔨 Query decomposer initialized")
    
    def decompose(self, query: str) -> List[Dict[str, Any]]:
        """
        Decompose complex query into sub-queries
        
        Args:
            query: Complex user query
            
        Returns:
            List of sub-query dictionaries
        """
        sub_queries = []
        
        # Detect if query has multiple parts
        if self._is_complex_query(query):
            # Split by conjunctions
            parts = self._split_by_conjunctions(query)
            
            for idx, part in enumerate(parts):
                sub_queries.append({
                    "query": part.strip(),
                    "order": idx,
                    "type": self._classify_subquery(part),
                    "parent_query": query
                })
        else:
            # Single simple query
            sub_queries.append({
                "query": query,
                "order": 0,
                "type": "simple",
                "parent_query": query
            })
        
        return sub_queries
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex"""
        complexity_indicators = [
            ' and ', ' or ', ' then ', ' also ',
            ' compare ', ' versus ', ' vs ',
            ', ', ';'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in complexity_indicators)
    
    def _split_by_conjunctions(self, query: str) -> List[str]:
        """Split query by conjunctions"""
        # Split by common separators
        parts = re.split(r'\s+and\s+|\s+or\s+|,|;', query, flags=re.IGNORECASE)
        
        # Clean and filter
        parts = [p.strip() for p in parts if p.strip()]
        
        return parts
    
    def _classify_subquery(self, subquery: str) -> str:
        """Classify sub-query type"""
        subquery_lower = subquery.lower()
        
        if any(word in subquery_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return "comparison"
        elif any(word in subquery_lower for word in ['total', 'sum', 'average', 'count']):
            return "aggregation"
        elif any(word in subquery_lower for word in ['show', 'list', 'find', 'get']):
            return "retrieval"
        else:
            return "analysis"
    
    def merge_results(
        self,
        sub_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge results from sub-queries
        
        Args:
            sub_results: List of results from each sub-query
            
        Returns:
            Merged results
        """
        all_docs = []
        all_metas = []
        all_scores = []
        
        for result in sub_results:
            all_docs.extend(result.get("documents", []))
            all_metas.extend(result.get("metadatas", []))
            all_scores.extend(result.get("scores", []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        unique_metas = []
        unique_scores = []
        
        for doc, meta, score in zip(all_docs, all_metas, all_scores):
            doc_id = doc[:50]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
                unique_metas.append(meta)
                unique_scores.append(score)
        
        return {
            "documents": unique_docs,
            "metadatas": unique_metas,
            "scores": unique_scores,
            "num_subqueries": len(sub_results)
        }


class IterativeRefiner:
    """
    Iteratively refines retrieval based on initial results
    """
    
    def __init__(self, max_iterations: int = 3):
        """
        Initialize iterative refiner
        
        Args:
            max_iterations: Maximum refinement iterations
        """
        self.max_iterations = max_iterations
        
        print(f"🔄 Iterative refiner initialized")
        print(f"   Max iterations: {max_iterations}")
    
    def refine_retrieval(
        self,
        original_query: str,
        initial_results: Dict[str, Any],
        retriever_func: callable,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Iteratively refine retrieval
        
        Args:
            original_query: Original query
            initial_results: Initial retrieval results
            retriever_func: Retrieval function
            n_results: Number of final results
            
        Returns:
            Refined results
        """
        current_results = initial_results
        all_results = [initial_results]
        
        for iteration in range(self.max_iterations):
            # Check if results are good enough
            if self._is_satisfied(current_results, n_results):
                break
            
            # Generate refinement query
            refinement_query = self._generate_refinement(
                original_query,
                current_results,
                iteration
            )
            
            # Retrieve with refined query
            try:
                refined_results = retriever_func(refinement_query, n_results=n_results)
                all_results.append(refined_results)
                
                # Merge with previous results
                current_results = self._merge_iterations(all_results)
                
            except Exception as e:
                print(f"   ⚠️ Refinement iteration {iteration + 1} failed: {e}")
                break
        
        return current_results
    
    def _is_satisfied(
        self,
        results: Dict[str, Any],
        target: int
    ) -> bool:
        """Check if results are satisfactory"""
        docs = results.get("documents", [])
        scores = results.get("scores", [])
        
        # Satisfied if we have enough high-quality results
        if len(docs) < target:
            return False
        
        # Check average score
        if scores:
            avg_score = sum(scores[:target]) / target
            return avg_score > 0.7  # High confidence threshold
        
        return True
    
    def _generate_refinement(
        self,
        original_query: str,
        current_results: Dict[str, Any],
        iteration: int
    ) -> str:
        """Generate refinement query"""
        docs = current_results.get("documents", [])
        
        if not docs:
            return original_query
        
        # Extract terms from top result
        top_doc = docs[0] if docs else ""
        doc_terms = set(top_doc.lower().split()[:20])
        
        # Extract terms from original query
        query_terms = set(original_query.lower().split())
        
        # Find new terms from documents
        new_terms = list(doc_terms - query_terms - {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to'
        })
        
        if new_terms:
            # Add new term to query
            refinement = f"{original_query} {new_terms[0]}"
        else:
            # Rephrase query
            refinement = f"more details about {original_query}"
        
        return refinement
    
    def _merge_iterations(
        self,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge results from all iterations"""
        merged_docs = []
        merged_metas = []
        merged_scores = []
        seen = set()
        
        # Process in reverse order (latest first)
        for results in reversed(all_results):
            for doc, meta, score in zip(
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("scores", [1.0] * len(results.get("documents", [])))
            ):
                doc_id = doc[:50]
                if doc_id not in seen:
                    seen.add(doc_id)
                    merged_docs.append(doc)
                    merged_metas.append(meta)
                    merged_scores.append(score)
        
        return {
            "documents": merged_docs,
            "metadatas": merged_metas,
            "scores": merged_scores,
            "iterations": len(all_results)
        }


class EnsembleRetriever:
    """
    Combines multiple retrieval strategies
    """
    
    def __init__(self):
        """Initialize ensemble retriever"""
        self.strategies = []
        
        print("🎭 Ensemble retriever initialized")
    
    def add_strategy(
        self,
        name: str,
        retriever_func: callable,
        weight: float = 1.0
    ) -> None:
        """
        Add retrieval strategy to ensemble
        
        Args:
            name: Strategy name
            retriever_func: Retrieval function
            weight: Strategy weight for voting
        """
        self.strategies.append({
            "name": name,
            "func": retriever_func,
            "weight": weight
        })
        
        print(f"   ✅ Added strategy: {name} (weight: {weight})")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve using all strategies and combine
        
        Args:
            query: Search query
            n_results: Number of final results
            
        Returns:
            Combined results
        """
        all_results = []
        
        # Retrieve with each strategy
        for strategy in self.strategies:
            try:
                results = strategy["func"](query, n_results=n_results * 2)
                
                # Add strategy info
                for doc, meta, score in zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                    results.get("scores", [1.0] * len(results.get("documents", [])))
                ):
                    all_results.append({
                        "document": doc,
                        "metadata": meta,
                        "score": score * strategy["weight"],
                        "strategy": strategy["name"]
                    })
                    
            except Exception as e:
                print(f"   ⚠️ Strategy {strategy['name']} failed: {e}")
                continue
        
        # Combine results using weighted voting
        combined = self._weighted_voting(all_results, n_results)
        
        return combined
    
    def _weighted_voting(
        self,
        all_results: List[Dict[str, Any]],
        n_results: int
    ) -> Dict[str, Any]:
        """Combine results using weighted voting"""
        # Group by document
        doc_scores = defaultdict(lambda: {
            "score": 0.0,
            "count": 0,
            "metadata": None,
            "strategies": []
        })
        
        for result in all_results:
            doc_id = result["document"][:50]
            doc_scores[doc_id]["score"] += result["score"]
            doc_scores[doc_id]["count"] += 1
            doc_scores[doc_id]["strategies"].append(result["strategy"])
            
            if doc_scores[doc_id]["metadata"] is None:
                doc_scores[doc_id]["metadata"] = result["metadata"]
                doc_scores[doc_id]["full_doc"] = result["document"]
        
        # Calculate average scores
        for doc_id in doc_scores:
            count = doc_scores[doc_id]["count"]
            doc_scores[doc_id]["avg_score"] = doc_scores[doc_id]["score"] / count
        
        # Sort by average score
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["avg_score"],
            reverse=True
        )[:n_results]
        
        # Format results
        docs = [item[1]["full_doc"] for item in sorted_results]
        metas = [item[1]["metadata"] for item in sorted_results]
        scores = [item[1]["avg_score"] for item in sorted_results]
        strategies = [", ".join(item[1]["strategies"]) for item in sorted_results]
        
        return {
            "documents": docs,
            "metadatas": metas,
            "scores": scores,
            "strategies_used": strategies,
            "num_strategies": len(self.strategies)
        }
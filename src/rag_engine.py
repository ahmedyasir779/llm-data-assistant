from typing import Dict, Any, List, Optional
from .enhanced_llm_client import EnhancedLLMClient
from .vector_store_advanced import AdvancedVectorStore
from .query_classifier import QueryClassifier, QueryRewriter
from .hybrid_search import HybridSearchEngine, ResultReranker, SearchRouter

from .token_manager import TokenManager, ContextWindow
from .context_compressor import ContextCompressor, RelevanceFilter

import pandas as pd


class RAGQueryEngine:
    """
    Retrieval-Augmented Generation engine for data analysis
    """
    
    def __init__(
        self,
        llm_client: Optional[EnhancedLLMClient] = None,
        vector_store: Optional[AdvancedVectorStore] = None
    ):
        """
        Initialize RAG engine
        
        Args:
            llm_client: LLM client instance
            vector_store: Vector store instance
        """
        self.llm = llm_client or EnhancedLLMClient()
        self.vector_store = vector_store or AdvancedVectorStore()
        
        self.classifier = QueryClassifier()
        self.rewriter = QueryRewriter()
        self.hybrid_engine = HybridSearchEngine(alpha=0.5)
        self.reranker = ResultReranker()
        self.router = SearchRouter(self.hybrid_engine, self.reranker)

        self.token_manager = TokenManager(max_tokens=4096)
        self.compressor = ContextCompressor(compression_ratio=0.6)
        self.relevance_filter = RelevanceFilter(min_relevance=0.3)
        self.context_window = ContextWindow(max_messages=10, max_tokens=4096)

        print("✅ RAG Query Engine initialized")
    
    def query_with_rag(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        n_context: int = 3,
        use_hybrid: bool = True,
        optimize_context: bool = True
    ) -> str:
        """Answer question using RAG with intelligent routing and optimization"""
        
        # Step 1: Classify query
        classification = self.classifier.classify_query(question)
        
        # Step 2: Retrieve with enhanced context
        semantic_results = self.vector_store.semantic_search(
            question,
            n_results=n_context * 3  # Get more for filtering
        )
        
        # Step 3: Optimize context if enabled
        if optimize_context:
            # Filter by relevance
            filtered_docs, filtered_metas, scores = self.relevance_filter.filter_by_relevance(
                semantic_results["documents"],
                semantic_results["metadatas"],
                question,
                semantic_results.get("distances")
            )
            
            # Remove duplicates
            filtered_docs = self.compressor.remove_duplicates(filtered_docs)
            
            # Compress to target size
            final_docs = self.compressor.compress_context(
                filtered_docs,
                question,
                max_docs=n_context
            )
            
            # Rebuild semantic results
            semantic_results = {
                "documents": final_docs,
                "metadatas": filtered_metas[:len(final_docs)],
                "distances": [1.0 - s for s in scores[:len(final_docs)]]
            }
        
        # Step 4: Route and search
        if use_hybrid and self.hybrid_engine.bm25_index:
            final_results = self.router.route_and_search(
                question,
                classification,
                semantic_results,
                n_results=n_context
            )
            
            context = "\n\n".join([
                f"[{meta['type']}] {doc}"
                for doc, meta in zip(final_results["documents"], final_results["metadatas"])
            ])
        else:
            context = self.vector_store.get_enhanced_context(question, n_results=n_context)
        
        # Step 5: Get sample data
        sample_data = self._get_relevant_samples(question, datasets, max_rows=15)
        
        # Step 6: Optimize token allocation
        system_prompt = """You are an advanced data analysis assistant with RAG capabilities.

            INSTRUCTIONS:
            1. Use the RETRIEVED CONTEXT to understand the data structure
            2. Use the SAMPLE DATA for specific value questions
            3. Provide accurate, specific answers with numbers
            4. Be concise but comprehensive"""

        if optimize_context:
            optimized = self.token_manager.optimize_context_allocation(
                system_prompt=system_prompt,
                context=context,
                sample_data=sample_data,
                question=question
            )
            
            context = optimized["context"]
            sample_data = optimized["sample_data"]
            system_prompt = optimized["system_prompt"]
        
        # Step 7: Create prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""CONTEXT:\n{context}\n\nDATA:\n{sample_data}\n\nQUESTION: {question}"""
            }
        ]
        
        # Step 8: Manage context window
        messages = self.context_window.manage_messages(messages)
        
        # Step 9: Generate response
        response = self.llm.chat(messages, temperature=0.3)
        
        return response
    def query_without_rag(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Answer question without RAG (direct data access)
        
        Args:
            question: User's question
            datasets: Dictionary of loaded datasets
            
        Returns:
            Generated answer
        """
        # Get comprehensive data info
        all_data = self._format_all_datasets(datasets)
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert data analyst. Analyze the data and provide specific answers based on actual values."""
            },
            {
                "role": "user",
                "content": f"""Here is the complete dataset:

{all_data}

Question: {question}

Provide a specific answer based on the data above."""
            }
        ]
        
        response = self.llm.chat(messages, temperature=0.3)
        return response
    
    def _create_rag_prompt(
        self,
        question: str,
        context: str,
        sample_data: str,
        classification: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Create optimized RAG prompt with classification"""
        
        system_prompt = """You are an advanced data analysis assistant with RAG capabilities.

            INSTRUCTIONS:
            1. Use the RETRIEVED CONTEXT to understand the data structure
            2. Use the SAMPLE DATA for specific value questions
            3. Provide accurate, specific answers with numbers
            4. If asked about trends, reference actual values
            5. Be concise but comprehensive

            FORMAT:
            - Start with direct answer
            - Support with specific data points
            - Use bullet points for clarity
            - Suggest visualizations when appropriate"""

        # Add classification hints if available
        if classification:
            if classification.get("requires_aggregation"):
                system_prompt += "\n- This query requires calculation/aggregation"
            if classification.get("requires_filtering"):
                system_prompt += "\n- This query requires filtering data"

        user_prompt = f"""RETRIEVED CONTEXT (from vector store):
            {context}

            SAMPLE DATA (actual rows):
            {sample_data}

            USER QUESTION: {question}

            Analyze the above information and provide a detailed answer."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _get_relevant_samples(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        max_rows: int = 20
    ) -> str:
        """Get relevant data samples based on question"""
        parts = []
        
        for name, df in datasets.items():
            parts.append(f"\n📄 {name} (showing {min(max_rows, len(df))} rows):")
            parts.append(df.head(max_rows).to_string())
        
        return "\n".join(parts)
    
    def _format_all_datasets(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Format all datasets for prompt"""
        parts = []
        
        for name, df in datasets.items():
            parts.append(f"\n{'='*60}")
            parts.append(f"📄 Dataset: {name}")
            parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            parts.append(f"Columns: {', '.join(df.columns)}")
            parts.append(f"\nFirst 10 rows:")
            parts.append(df.head(10).to_string())
            
            # Add stats if numeric columns exist
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                parts.append(f"\nStatistics:")
                parts.append(df[numeric_cols].describe().to_string())
        
        return "\n".join(parts)
    
    def compare_rag_vs_direct(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, str]:
        """
        Compare RAG answer vs direct answer
        Useful for debugging and optimization
        """
        print("🔍 Comparing RAG vs Direct...")
        
        rag_answer = self.query_with_rag(question, datasets)
        direct_answer = self.query_without_rag(question, datasets)
        
        return {
            "rag_answer": rag_answer,
            "direct_answer": direct_answer,
            "question": question
        }
from typing import Dict, Any, List, Optional
from .enhanced_llm_client import EnhancedLLMClient
from .vector_store_advanced import AdvancedVectorStore
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
        
        print("✅ RAG Query Engine initialized")
    
    def query_with_rag(
        self,
        question: str,
        datasets: Dict[str, pd.DataFrame],
        n_context: int = 3,
        use_hybrid: bool = True
    ) -> str:
        """
        Answer question using RAG
        
        Args:
            question: User's question
            datasets: Dictionary of loaded datasets
            n_context: Number of context pieces to retrieve
            use_hybrid: Whether to use hybrid search
            
        Returns:
            Generated answer
        """
        # Step 1: Retrieve relevant context
        context = self.vector_store.get_enhanced_context(
            question,
            n_results=n_context,
            include_keywords=use_hybrid
        )
        
        # Step 2: Get sample data for specific questions
        sample_data = self._get_relevant_samples(question, datasets)
        
        # Step 3: Create comprehensive prompt
        messages = self._create_rag_prompt(question, context, sample_data)
        
        # Step 4: Generate response
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
        sample_data: str
    ) -> List[Dict[str, str]]:
        """Create optimized RAG prompt"""
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
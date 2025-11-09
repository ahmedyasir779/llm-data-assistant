from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from pathlib import Path

# Import all components
from .enhanced_llm_client import EnhancedLLMClient
from .vector_store_advanced import AdvancedVectorStore
from .rag_engine import RAGQueryEngine
from .query_classifier import QueryClassifier
from .hybrid_search import HybridSearchEngine, SearchRouter, ResultReranker
from .advanced_retrieval import MultiQueryRetriever, EnsembleRetriever
from .token_manager import TokenManager
from .context_compressor import ContextCompressor, RelevanceFilter
from .embedding_manager import EmbeddingModelManager
from .error_handler import ErrorHandler, RetryHandler
from .monitoring import PerformanceMonitor, ApplicationLogger
from .config import ConfigManager


class LLMDataAssistant:
    """
    Complete integrated LLM Data Assistant
    Production-ready with all optimizations
    """
    
    def __init__(self, config_file: str = ".env"):
        """
        Initialize complete system
        
        Args:
            config_file: Configuration file path
        """
        print("\n" + "="*60)
        print("🚀 INITIALIZING LLM DATA ASSISTANT v2.4.0")
        print("="*60 + "\n")
        
        # Load configuration
        print("⚙️ Loading configuration...")
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.load_config()
        
        # Initialize monitoring
        print("📊 Setting up monitoring...")
        self.performance_monitor = PerformanceMonitor(
            history_size=self.config.monitoring.metrics_history_size
        )
        self.logger = ApplicationLogger(
            name="llm-data-assistant",
            log_level=self.config.monitoring.log_level
        )
        self.error_handler = ErrorHandler(
            log_errors=self.config.monitoring.enable_logging
        )
        
        # Initialize core components
        print("🧠 Initializing AI components...")
        self.llm_client = EnhancedLLMClient(
            model=self.config.llm.model,
            max_retries=self.config.performance.max_retries
        )
        
        self.vector_store = AdvancedVectorStore(
            collection_name=self.config.vector_store.collection_name,
            persist_directory=self.config.vector_store.persist_directory,
            embedding_model=self.config.vector_store.embedding_model
        )
        
        # Initialize optimization components
        print("⚡ Setting up optimizations...")
        self.token_manager = TokenManager(max_tokens=self.config.llm.max_tokens)
        self.compressor = ContextCompressor(
            compression_ratio=self.config.performance.compression_ratio
        )
        self.relevance_filter = RelevanceFilter(
            min_relevance=self.config.search.min_relevance
        )
        
        # Initialize search components
        print("🔍 Configuring search...")
        self.query_classifier = QueryClassifier()
        self.multi_query = MultiQueryRetriever(num_queries=3)
        self.hybrid_engine = HybridSearchEngine(alpha=self.config.search.alpha)
        self.reranker = ResultReranker()
        self.search_router = SearchRouter(self.hybrid_engine, self.reranker)
        
        # Initialize RAG engine
        print("🤖 Building RAG engine...")
        self.rag_engine = RAGQueryEngine(
            llm_client=self.llm_client,
            vector_store=self.vector_store
        )
        
        # State
        self.datasets = {}
        
        print("\n" + "="*60)
        print("✅ LLM DATA ASSISTANT READY!")
        print("="*60 + "\n")
    
    def add_dataset(self, file_path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Add dataset to the system
        
        Args:
            file_path: Path to data file
            name: Optional dataset name
            
        Returns:
            Dataset information
        """
        start_time = self.performance_monitor.start_query()
        
        try:
            # Load data
            path = Path(file_path)
            if not name:
                name = path.name
            
            if path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            
            # Store dataset
            self.datasets[name] = df
            
            # Index in vector store
            num_docs = self.vector_store.add_dataframe_context(
                df, name, chunk_size=self.config.vector_store.chunk_size
            )
            
            # Update hybrid search index
            documents = [f"Dataset: {name}"] * len(df)
            metadatas = [{"file": name, "type": "data"}] * len(df)
            self.hybrid_engine.index_documents(documents, metadatas)
            
            self.performance_monitor.end_query(start_time, success=True)
            
            self.logger.log_system_event("dataset_added", {
                "name": name,
                "rows": len(df),
                "columns": len(df.columns),
                "documents": num_docs
            })
            
            return {
                "name": name,
                "rows": len(df),
                "columns": len(df.columns),
                "documents_indexed": num_docs,
                "status": "success"
            }
            
        except Exception as e:
            self.performance_monitor.end_query(start_time, success=False)
            self.error_handler.handle_error(e, "add_dataset")
            raise
    
    def query(
        self,
        question: str,
        use_advanced: bool = True
    ) -> Dict[str, Any]:
        """
        Query the system
        
        Args:
            question: User question
            use_advanced: Use advanced retrieval strategies
            
        Returns:
            Response with metadata
        """
        start_time = self.performance_monitor.start_query()
        
        try:
            # Classify query
            classification = self.query_classifier.classify_query(question)
            
            # Generate query variations if advanced
            queries = [question]
            if use_advanced:
                queries = self.multi_query.generate_queries(question)
            
            # Get RAG response with all optimizations
            response = self.rag_engine.query_with_rag(
                question,
                self.datasets,
                n_context=self.config.search.n_results,
                use_hybrid=self.config.search.use_hybrid,
                optimize_context=True
            )
            
            duration = self.performance_monitor.end_query(start_time, success=True)
            
            self.logger.log_query(question, duration, success=True)
            
            return {
                "question": question,
                "answer": response,
                "query_type": classification["query_type"].value,
                "duration_s": duration,
                "queries_generated": len(queries),
                "status": "success"
            }
            
        except Exception as e:
            self.performance_monitor.end_query(start_time, success=False)
            self.error_handler.handle_error(e, "query")
            self.logger.log_error("query", e)
            
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "status": "error"
            }
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        health = self.performance_monitor.get_health_status()
        
        # Add component status
        health["components"] = {
            "llm": "operational",
            "vector_store": "operational",
            "datasets_loaded": len(self.datasets)
        }
        
        return health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        perf_stats = self.performance_monitor.get_performance_stats()
        error_summary = self.error_handler.get_error_summary()
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "performance": perf_stats,
            "errors": error_summary,
            "vector_store": vector_stats,
            "datasets": {
                "count": len(self.datasets),
                "names": list(self.datasets.keys())
            }
        }
    
    def print_dashboard(self) -> None:
        """Print system dashboard"""
        health = self.get_health()
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📊 LLM DATA ASSISTANT DASHBOARD")
        print("="*70)
        
        print(f"\n🏥 Health: {health['status'].upper()}")
        if health['issues']:
            print(f"   ⚠️  Issues: {', '.join(health['issues'])}")
        
        print(f"\n📈 Performance:")
        perf = stats['performance']
        print(f"   Total queries: {perf['total_queries']}")
        print(f"   Success rate: {perf.get('success_rate', 100):.1f}%")
        if 'avg_query_time_s' in perf:
            print(f"   Avg response: {perf['avg_query_time_s']:.2f}s")
        
        print(f"\n🗄️ Vector Store:")
        print(f"   Documents: {stats['vector_store']['total_documents']:,}")
        print(f"   Model: {stats['vector_store']['embedding_model']}")
        
        print(f"\n📊 Datasets:")
        print(f"   Loaded: {stats['datasets']['count']}")
        if stats['datasets']['names']:
            print(f"   Names: {', '.join(stats['datasets']['names'][:3])}")
        
        print(f"\n⏱️  Uptime: {health['uptime_seconds']/60:.1f} minutes")
        
        print("\n" + "="*70 + "\n")
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import os
from pathlib import Path


class LLMConfig(BaseModel):
    """LLM configuration"""
    api_key: str = Field(..., description="API key for LLM service")
    model: str = Field(default="llama-3.1-8b-instant", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Max tokens")
    timeout: int = Field(default=30, ge=1, le=120, description="Timeout in seconds")
    
    @validator('api_key')
    def api_key_not_empty(cls, v):
        if not v or v == "":
            raise ValueError("API key cannot be empty")
        return v


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    collection_name: str = Field(default="data_assistant", description="Collection name")
    persist_directory: str = Field(default="./chroma_db", description="Persist directory")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    chunk_size: int = Field(default=50, ge=1, le=1000, description="Chunk size")


class SearchConfig(BaseModel):
    """Search configuration"""
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic vs keyword weight")
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0, description="Min relevance score")


class PerformanceConfig(BaseModel):
    """Performance configuration"""
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    enable_cache: bool = Field(default=True, description="Enable caching")
    max_cache_size_gb: int = Field(default=2, ge=1, le=10, description="Max cache size")
    compression_ratio: float = Field(default=0.6, ge=0.1, le=1.0, description="Compression ratio")


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enable_logging: bool = Field(default=True, description="Enable logging")
    log_level: str = Field(default="INFO", description="Log level")
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    metrics_history_size: int = Field(default=100, ge=10, le=1000, description="Metrics history")


class ApplicationConfig(BaseModel):
    """Complete application configuration"""
    llm: LLMConfig
    vector_store: VectorStoreConfig
    search: SearchConfig
    performance: PerformanceConfig
    monitoring: MonitoringConfig
    
    class Config:
        arbitrary_types_allowed = True


class ConfigManager:
    """
    Configuration manager with environment variable support
    """
    
    def __init__(self, env_file: Optional[str] = ".env"):
        """
        Initialize config manager
        
        Args:
            env_file: Path to .env file
        """
        self.env_file = env_file
        
        # Load environment variables
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
        
        print("⚙️ Configuration manager initialized")
    
    def load_config(self) -> ApplicationConfig:
        """Load configuration from environment"""
        config = ApplicationConfig(
            llm=LLMConfig(
                api_key=os.getenv("GROQ_API_KEY", ""),
                model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
                timeout=int(os.getenv("TIMEOUT", "30"))
            ),
            vector_store=VectorStoreConfig(
                collection_name=os.getenv("COLLECTION_NAME", "data_assistant"),
                persist_directory=os.getenv("PERSIST_DIR", "./chroma_db"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "50"))
            ),
            search=SearchConfig(
                n_results=int(os.getenv("N_RESULTS", "5")),
                use_hybrid=os.getenv("USE_HYBRID", "true").lower() == "true",
                alpha=float(os.getenv("ALPHA", "0.5")),
                min_relevance=float(os.getenv("MIN_RELEVANCE", "0.3"))
            ),
            performance=PerformanceConfig(
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
                max_cache_size_gb=int(os.getenv("MAX_CACHE_GB", "2")),
                compression_ratio=float(os.getenv("COMPRESSION_RATIO", "0.6"))
            ),
            monitoring=MonitoringConfig(
                enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
                metrics_history_size=int(os.getenv("METRICS_HISTORY", "100"))
            )
        )
        
        return config
    
    def save_config(self, config: ApplicationConfig, filename: str = "config.json") -> None:
        """Save configuration to file"""
        import json
        
        config_dict = config.dict()
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved to {filename}")
    
    def print_config(self, config: ApplicationConfig) -> None:
        """Print current configuration"""
        print("\n" + "="*60)
        print("⚙️ CURRENT CONFIGURATION")
        print("="*60)
        
        print("\n🤖 LLM:")
        print(f"   Model: {config.llm.model}")
        print(f"   Temperature: {config.llm.temperature}")
        print(f"   Max tokens: {config.llm.max_tokens}")
        
        print("\n🗄️ Vector Store:")
        print(f"   Collection: {config.vector_store.collection_name}")
        print(f"   Embedding model: {config.vector_store.embedding_model}")
        
        print("\n🔍 Search:")
        print(f"   Results: {config.search.n_results}")
        print(f"   Hybrid search: {config.search.use_hybrid}")
        print(f"   Alpha: {config.search.alpha}")
        
        print("\n⚡ Performance:")
        print(f"   Max retries: {config.performance.max_retries}")
        print(f"   Cache enabled: {config.performance.enable_cache}")
        
        print("\n📊 Monitoring:")
        print(f"   Logging: {config.monitoring.enable_logging}")
        print(f"   Log level: {config.monitoring.log_level}")
        
        print("="*60 + "\n")
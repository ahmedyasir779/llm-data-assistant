from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import time
from diskcache import Cache
import hashlib


class EmbeddingModelManager:
    """
    Manages multiple embedding models with performance optimization
    """
    
    # Available embedding models (ordered by performance/size trade-off)
    AVAILABLE_MODELS = {
        "mini": {
            "name": "all-MiniLM-L6-v2",
            "dim": 384,
            "size_mb": 80,
            "speed": "fast",
            "quality": "good",
            "description": "Fast and efficient, good for most tasks"
        },
        "small": {
            "name": "all-mpnet-base-v2",
            "dim": 768,
            "size_mb": 420,
            "speed": "medium",
            "quality": "excellent",
            "description": "Best quality, slower but more accurate"
        },
        "multilingual": {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "dim": 384,
            "size_mb": 420,
            "speed": "medium",
            "quality": "good",
            "description": "Supports 50+ languages including Arabic"
        },
        "fast": {
            "name": "all-MiniLM-L12-v2",
            "dim": 384,
            "size_mb": 120,
            "speed": "very-fast",
            "quality": "good",
            "description": "Ultra-fast for real-time applications"
        }
    }
    
    def __init__(
        self,
        model_type: str = "mini",
        cache_dir: str = "./embedding_cache",
        enable_cache: bool = True
    ):
        """
        Initialize embedding manager
        
        Args:
            model_type: Type of model to use (mini/small/multilingual/fast)
            cache_dir: Directory for embedding cache
            enable_cache: Whether to enable caching
        """
        self.model_type = model_type
        
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_info = self.AVAILABLE_MODELS[model_type]
        self.model_name = self.model_info["name"]
        
        # Initialize model
        print(f"🧠 Loading embedding model: {self.model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(self.model_name)
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.2f}s")
        
        # Setup cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache = Cache(str(self.cache_dir))
            print(f"   💾 Cache enabled: {self.cache_dir}")
        else:
            self.cache = None
        
        # Performance tracking
        self.stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0
        }
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings with caching
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for idx, text in enumerate(texts):
            if self.enable_cache:
                cached = self._get_from_cache(text)
                if cached is not None:
                    embeddings.append((idx, cached))
                    self.stats["cache_hits"] += 1
                    continue
            
            texts_to_encode.append(text)
            indices_to_encode.append(idx)
            self.stats["cache_misses"] += 1
        
        # Encode uncached texts
        if texts_to_encode:
            start_time = time.time()
            
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            encode_time = time.time() - start_time
            self.stats["total_time"] += encode_time
            
            # Cache new embeddings
            if self.enable_cache:
                for text, embedding in zip(texts_to_encode, new_embeddings):
                    self._save_to_cache(text, embedding)
            
            # Add to results
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original indices and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        self.stats["total_embeddings"] += len(texts)
        
        return result
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (convenience method)"""
        return self.encode([text])[0]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Include model name in key to avoid conflicts
        key_text = f"{self.model_name}:{text}"
        return hashlib.md5(key_text.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if not self.cache:
            return None
        
        key = self._get_cache_key(text)
        cached = self.cache.get(key)
        
        if cached is not None:
            return np.array(cached)
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache"""
        if not self.cache:
            return
        
        key = self._get_cache_key(text)
        self.cache.set(key, embedding.tolist())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests * 100
            if total_requests > 0 else 0
        )
        
        avg_time = (
            self.stats["total_time"] / self.stats["cache_misses"]
            if self.stats["cache_misses"] > 0 else 0
        )
        
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "embedding_dim": self.model_info["dim"],
            "total_embeddings": self.stats["total_embeddings"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_time": f"{self.stats['total_time']:.2f}s",
            "avg_time_per_batch": f"{avg_time:.3f}s"
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear()
            print("✅ Cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return self.model_info.copy()
    
    @classmethod
    def list_available_models(cls) -> None:
        """Print all available models"""
        print("\n📊 Available Embedding Models:")
        print("="*70)
        
        for key, info in cls.AVAILABLE_MODELS.items():
            print(f"\n🔹 {key.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Dimensions: {info['dim']}")
            print(f"   Size: {info['size_mb']} MB")
            print(f"   Speed: {info['speed']}")
            print(f"   Quality: {info['quality']}")
            print(f"   Description: {info['description']}")
        
        print("\n" + "="*70 + "\n")


class EmbeddingBenchmark:
    """
    Benchmark different embedding models for performance comparison
    """
    
    def __init__(self, test_texts: Optional[List[str]] = None):
        """
        Initialize benchmark
        
        Args:
            test_texts: Optional custom test texts
        """
        self.test_texts = test_texts or self._generate_test_texts()
    
    def _generate_test_texts(self) -> List[str]:
        """Generate diverse test texts"""
        return [
            "What is the total revenue for Q4?",
            "Show me products with high ratings",
            "Calculate average sales per region",
            "Which customers have the highest value?",
            "Compare this year's performance to last year",
            "Find all transactions above $1000",
            "What are the top 10 selling products?",
            "Show me the distribution of customer ages",
            "Identify products with declining sales",
            "What is the correlation between price and sales?",
        ]
    
    def benchmark_model(
        self,
        model_type: str,
        enable_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark a specific model
        
        Args:
            model_type: Model type to benchmark
            enable_cache: Whether to enable caching
            
        Returns:
            Benchmark results
        """
        print(f"\n🔍 Benchmarking: {model_type}")
        
        # Initialize model
        manager = EmbeddingModelManager(
            model_type=model_type,
            enable_cache=enable_cache
        )
        
        # First run (cold cache)
        print("   Running cold cache test...")
        start_time = time.time()
        embeddings1 = manager.encode(self.test_texts)
        cold_time = time.time() - start_time
        
        # Second run (warm cache)
        print("   Running warm cache test...")
        start_time = time.time()
        embeddings2 = manager.encode(self.test_texts)
        warm_time = time.time() - start_time
        
        # Get stats
        stats = manager.get_stats()
        model_info = manager.get_model_info()
        
        results = {
            "model_type": model_type,
            "model_name": model_info["name"],
            "embedding_dim": model_info["dim"],
            "model_size_mb": model_info["size_mb"],
            "speed_rating": model_info["speed"],
            "quality_rating": model_info["quality"],
            "cold_time": f"{cold_time:.3f}s",
            "warm_time": f"{warm_time:.3f}s",
            "speedup": f"{cold_time/warm_time if warm_time > 0 else 0:.1f}x",
            "cache_hit_rate": stats["cache_hit_rate"],
            "embeddings_shape": embeddings1.shape
        }
        
        return results
    
    def compare_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare all available models
        
        Returns:
            Dictionary of benchmark results
        """
        print("\n" + "="*70)
        print("🏁 EMBEDDING MODEL COMPARISON")
        print("="*70)
        
        results = {}
        
        for model_type in EmbeddingModelManager.AVAILABLE_MODELS.keys():
            try:
                results[model_type] = self.benchmark_model(model_type)
            except Exception as e:
                print(f"   ❌ Error benchmarking {model_type}: {e}")
                results[model_type] = {"error": str(e)}
        
        # Print comparison table
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print formatted comparison table"""
        print("\n" + "="*70)
        print("📊 BENCHMARK RESULTS")
        print("="*70)
        
        print(f"\n{'Model':<15} {'Size':<10} {'Speed':<12} {'Cold':<10} {'Warm':<10} {'Speedup':<10}")
        print("-"*70)
        
        for model_type, result in results.items():
            if "error" not in result:
                print(
                    f"{model_type:<15} "
                    f"{result['model_size_mb']:<10} "
                    f"{result['speed_rating']:<12} "
                    f"{result['cold_time']:<10} "
                    f"{result['warm_time']:<10} "
                    f"{result['speedup']:<10}"
                )
        
        print("\n" + "="*70)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-"*70)
        
        fastest = min(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: float(x[1]["warm_time"].rstrip("s"))
        )
        
        smallest = min(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1]["model_size_mb"]
        )
        
        print(f"🚀 Fastest model: {fastest[0]} ({fastest[1]['warm_time']})")
        print(f"📦 Smallest model: {smallest[0]} ({smallest[1]['model_size_mb']} MB)")
        print(f"⭐ Recommended: mini (best balance of speed/quality)")
        print(f"🌍 For Arabic: multilingual")
        print("\n" + "="*70 + "\n")


def recommend_model(
    use_case: str,
    dataset_size: str = "medium",
    language: str = "english"
) -> str:
    """
    Recommend best embedding model based on use case
    
    Args:
        use_case: "speed", "quality", "balanced", "multilingual"
        dataset_size: "small", "medium", "large"
        language: "english", "arabic", "multilingual"
        
    Returns:
        Recommended model type
    """
    print("\n🎯 MODEL RECOMMENDATION")
    print("="*50)
    print(f"Use case: {use_case}")
    print(f"Dataset size: {dataset_size}")
    print(f"Language: {language}")
    print("-"*50)
    
    # Decision logic
    if language in ["arabic", "multilingual"]:
        recommendation = "multilingual"
        reason = "Supports Arabic and 50+ languages"
    
    elif use_case == "speed" or dataset_size == "large":
        recommendation = "fast"
        reason = "Ultra-fast for large datasets"
    
    elif use_case == "quality":
        recommendation = "small"
        reason = "Best quality embeddings"
    
    else:  # balanced
        recommendation = "mini"
        reason = "Best balance of speed and quality"
    
    print(f"\n✅ Recommended: {recommendation}")
    print(f"💡 Reason: {reason}")
    print("="*50 + "\n")
    
    return recommendation
from diskcache import Cache
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import hashlib
import json
from pathlib import Path
import time


class EmbeddingCache:
    """
    Advanced embedding cache with multiple strategies
    """
    
    def __init__(
        self,
        cache_dir: str = "./embedding_cache",
        max_size_gb: int = 2,
        strategy: str = "lru"
    ):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            strategy: Caching strategy (lru/lfu/fifo)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Map strategy names to diskcache eviction policies
        strategy_map = {
            "lru": "least-recently-used",
            "lfu": "least-frequently-used", 
            "fifo": "least-recently-stored"
        }
        
        # Get actual eviction policy or default
        eviction_policy = strategy_map.get(strategy, "least-recently-used")
        
        # Initialize disk cache
        self.cache = Cache(
            str(self.cache_dir),
            size_limit=max_size_gb * 1024 * 1024 * 1024,  # Convert to bytes
            eviction_policy=eviction_policy
        )
        
        self.strategy = strategy
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0
        }
        
        print(f"💾 Embedding cache initialized")
        print(f"   Directory: {self.cache_dir}")
        print(f"   Max size: {max_size_gb} GB")
        print(f"   Strategy: {strategy} ({eviction_policy})")
    
    def get(self, key: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache
        
        Args:
            key: Cache key (text hash)
            model_name: Model name for namespacing
            
        Returns:
            Cached embedding or None
        """
        cache_key = f"{model_name}:{key}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            self.stats["hits"] += 1
            return np.array(cached)
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        key: str,
        embedding: np.ndarray,
        model_name: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Save embedding to cache
        
        Args:
            key: Cache key
            embedding: Embedding array
            model_name: Model name
            ttl: Time to live in seconds (optional)
        """
        cache_key = f"{model_name}:{key}"
        self.cache.set(
            cache_key,
            embedding.tolist(),
            expire=ttl
        )
        self.stats["saves"] += 1
    
    def get_batch(
        self,
        keys: List[str],
        model_name: str
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get batch of embeddings from cache
        
        Args:
            keys: List of cache keys
            model_name: Model name
            
        Returns:
            Tuple of (embeddings list, indices of missing keys)
        """
        embeddings = []
        missing_indices = []
        
        for idx, key in enumerate(keys):
            embedding = self.get(key, model_name)
            embeddings.append(embedding)
            
            if embedding is None:
                missing_indices.append(idx)
        
        return embeddings, missing_indices
    
    def set_batch(
        self,
        keys: List[str],
        embeddings: List[np.ndarray],
        model_name: str
    ) -> None:
        """
        Save batch of embeddings to cache
        
        Args:
            keys: List of cache keys
            embeddings: List of embedding arrays
            model_name: Model name
        """
        for key, embedding in zip(keys, embeddings):
            self.set(key, embedding, model_name)
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0
        }
        print("✅ Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests * 100
            if total_requests > 0 else 0
        )
        
        # Get cache info - returns tuple (hits, misses)
        try:
            # Try to get volume stats if available
            cache_size = self.cache.volume()
        except:
            cache_size = 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "saves": self.stats["saves"],
            "hit_rate": f"{hit_rate:.1f}%",
            "size_mb": cache_size / (1024 * 1024) if cache_size > 0 else 0,
            "items": len(self.cache),
            "strategy": self.strategy
        }
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize cache by removing least useful entries
        
        Returns:
            Optimization stats
        """
        print("🔧 Optimizing cache...")
        
        initial_size = len(self.cache)
        initial_stats = self.get_stats()
        
        # Cache automatically handles eviction based on strategy
        # We can trigger cleanup
        self.cache.cull()
        
        final_size = len(self.cache)
        removed = initial_size - final_size
        
        print(f"   ✅ Removed {removed} entries")
        print(f"   📦 Cache size: {final_size} items")
        
        return {
            "initial_size": initial_size,
            "final_size": final_size,
            "removed": removed,
            "size_mb": initial_stats["size_mb"]
        }


class SmartEmbeddingManager:
    """
    Smart embedding manager with automatic optimization
    Combines model management with intelligent caching
    """
    
    def __init__(
        self,
        model_type: str = "mini",
        enable_cache: bool = True,
        auto_optimize: bool = True
    ):
        """
        Initialize smart embedding manager
        
        Args:
            model_type: Type of embedding model
            enable_cache: Enable caching
            auto_optimize: Auto-optimize cache periodically
        """
        from embedding_manager import EmbeddingModelManager
        
        self.model_manager = EmbeddingModelManager(
            model_type=model_type,
            enable_cache=enable_cache
        )
        
        self.cache = EmbeddingCache() if enable_cache else None
        self.auto_optimize = auto_optimize
        self.encode_count = 0
        self.optimize_interval = 1000  # Optimize every N encodings
        
        print(f"🧠 Smart embedding manager initialized")
        print(f"   Auto-optimize: {auto_optimize}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode texts with smart caching
        
        Args:
            texts: Texts to encode
            batch_size: Batch size
            
        Returns:
            Embeddings array
        """
        embeddings = self.model_manager.encode(
            texts,
            batch_size=batch_size
        )
        
        self.encode_count += len(texts)
        
        # Auto-optimize if needed
        if (self.auto_optimize and 
            self.cache and 
            self.encode_count % self.optimize_interval == 0):
            self.cache.optimize()
        
        return embeddings
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        model_stats = self.model_manager.get_stats()
        cache_stats = self.cache.get_stats() if self.cache else {}
        
        return {
            "model": model_stats,
            "cache": cache_stats,
            "total_encoded": self.encode_count
        }
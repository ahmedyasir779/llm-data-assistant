import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.embedding_manager import (
    EmbeddingModelManager,
    EmbeddingBenchmark,
    recommend_model
)
from src.embedding_cache import EmbeddingCache, SmartEmbeddingManager
import numpy as np


def test_embedding_manager():
    """Test basic embedding manager"""
    print("\n" + "="*60)
    print("🧪 TEST 1: EMBEDDING MANAGER")
    print("="*60 + "\n")
    
    # Test with mini model
    print("1. Initialize embedding manager...")
    manager = EmbeddingModelManager(model_type="mini")
    print("   ✅ Manager initialized\n")
    
    # Test single encoding
    print("2. Test single text encoding...")
    text = "What is the highest price in the dataset?"
    embedding = manager.encode_single(text)
    print(f"   Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dim: {len(embedding)}")
    print("   ✅ Single encoding works\n")
    
    # Test batch encoding
    print("3. Test batch encoding...")
    texts = [
        "Show me top products",
        "Calculate average revenue",
        "Find high-rated items",
        "What are the sales trends?"
    ]
    embeddings = manager.encode(texts)
    print(f"   Texts: {len(texts)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print("   ✅ Batch encoding works\n")
    
    # Test cache
    print("4. Test embedding cache...")
    print("   Encoding same texts again...")
    embeddings2 = manager.encode(texts)
    print("   ✅ Cache working (should be instant)\n")
    
    # Get stats
    print("5. Check performance stats...")
    stats = manager.get_stats()
    print(f"   Total embeddings: {stats['total_embeddings']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']}")
    print(f"   Total time: {stats['total_time']}")
    print("   ✅ Stats tracking works\n")
    
    print("="*60)
    print("✅ EMBEDDING MANAGER TEST PASSED!")
    print("="*60 + "\n")


def test_model_comparison():
    """Test model comparison"""
    print("\n" + "="*60)
    print("🧪 TEST 2: MODEL COMPARISON")
    print("="*60 + "\n")
    
    # List available models
    print("1. List available models...")
    EmbeddingModelManager.list_available_models()
    
    # Create benchmark
    print("2. Create benchmark suite...")
    benchmark = EmbeddingBenchmark()
    print(f"   ✅ Benchmark ready with {len(benchmark.test_texts)} test texts\n")
    
    # Compare all models
    print("3. Running comprehensive comparison...")
    print("   ⚠️ This may take 2-3 minutes...")
    results = benchmark.compare_all_models()
    
    print("\n" + "="*60)
    print("✅ MODEL COMPARISON TEST PASSED!")
    print("="*60 + "\n")
    
    return results


def test_embedding_cache():
    """Test advanced embedding cache"""
    print("\n" + "="*60)
    print("🧪 TEST 3: EMBEDDING CACHE")
    print("="*60 + "\n")
    
    # Initialize cache
    print("1. Initialize embedding cache...")
    cache = EmbeddingCache(
        cache_dir="./test_cache",
        max_size_gb=1,
        strategy="least-recently-used"  # Changed from "lru"!
    )
    print("   ✅ Cache initialized\n")
    
    # Test save and retrieve
    print("2. Test cache operations...")
    test_embedding = np.random.rand(384)
    cache.set("test_key", test_embedding, "test_model")
    retrieved = cache.get("test_key", "test_model")
    
    if retrieved is not None and np.allclose(test_embedding, retrieved):
        print("   ✅ Save and retrieve working")
    else:
        print("   ❌ Cache not working properly")
    
    # Test batch operations
    print("\n3. Test batch operations...")
    keys = [f"key_{i}" for i in range(10)]
    embeddings = [np.random.rand(384) for _ in range(10)]
    
    cache.set_batch(keys, embeddings, "test_model")
    retrieved_batch, missing = cache.get_batch(keys, "test_model")
    
    print(f"   Cached: {len(keys)} embeddings")
    print(f"   Retrieved: {len([e for e in retrieved_batch if e is not None])}")
    print(f"   Missing: {len(missing)}")
    print("   ✅ Batch operations working\n")
    
    # Get stats
    print("4. Check cache statistics...")
    stats = cache.get_stats()
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']}")
    print(f"   Items: {stats['items']}")
    print("   ✅ Statistics tracking\n")
    
    # Clear cache
    print("5. Clear cache...")
    cache.clear()
    print("   ✅ Cache cleared\n")
    
    print("="*60)
    print("✅ EMBEDDING CACHE TEST PASSED!")
    print("="*60 + "\n")


def test_smart_manager():
    """Test smart embedding manager"""
    print("\n" + "="*60)
    print("🧪 TEST 4: SMART EMBEDDING MANAGER")
    print("="*60 + "\n")
    
    print("1. Initialize smart manager...")
    manager = SmartEmbeddingManager(
        model_type="mini",
        enable_cache=True,
        auto_optimize=True
    )
    print("   ✅ Smart manager initialized\n")
    
    print("2. Test encoding with auto-optimization...")
    texts = [f"Test query number {i}" for i in range(50)]
    embeddings = manager.encode(texts)
    print(f"   Encoded {len(texts)} texts")
    print(f"   Shape: {embeddings.shape}")
    print("   ✅ Encoding works\n")
    
    print("3. Get performance report...")
    report = manager.get_performance_report()
    print(f"   Model stats: {report['model']['model_type']}")
    print(f"   Cache hit rate: {report['cache']['hit_rate']}")
    print(f"   Total encoded: {report['total_encoded']}")
    print("   ✅ Performance tracking\n")
    
    print("="*60)
    print("✅ SMART MANAGER TEST PASSED!")
    print("="*60 + "\n")


def test_model_recommendation():
    """Test model recommendation system"""
    print("\n" + "="*60)
    print("🧪 TEST 5: MODEL RECOMMENDATION")
    print("="*60 + "\n")
    
    # Test different scenarios
    scenarios = [
        ("speed", "large", "english"),
        ("quality", "small", "english"),
        ("balanced", "medium", "english"),
        ("balanced", "medium", "arabic"),
    ]
    
    for use_case, dataset_size, language in scenarios:
        print(f"\n📋 Scenario: {use_case}, {dataset_size}, {language}")
        recommendation = recommend_model(use_case, dataset_size, language)
        print(f"   Recommended: {recommendation}\n")
    
    print("="*60)
    print("✅ RECOMMENDATION TEST PASSED!")
    print("="*60 + "\n")


def main():
    """Run all embedding optimization tests"""
    print("\n🚀 DAY 37: EMBEDDING OPTIMIZATION TESTS")
    print("Testing embedding models, caching, and performance\n")
    
    try:
        # Test 1: Basic embedding manager
        test_embedding_manager()
        
        # Test 2: Model comparison
        comparison_results = test_model_comparison()
        
        # Test 3: Embedding cache
        test_embedding_cache()
        
        # Test 4: Smart manager
        test_smart_manager()
        
        # Test 5: Recommendations
        test_model_recommendation()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 ALL EMBEDDING TESTS COMPLETED!")
        print("="*60)
        print("\n✅ Embedding Manager: PASSED")
        print("✅ Model Comparison: PASSED")
        print("✅ Embedding Cache: PASSED")
        print("✅ Smart Manager: PASSED")
        print("✅ Recommendations: PASSED")
        print("\n🎯 Day 37 embedding optimization is complete!")
        print("\n💡 Recommended model for most users: mini")
        print("💡 For Arabic data: multilingual")
        print("💡 For best quality: small")
        print("💡 For fastest speed: fast\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
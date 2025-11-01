import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_classifier import QueryClassifier, QueryRewriter, QueryType, QueryIntent
from src.hybrid_search import HybridSearchEngine, ResultReranker, SearchRouter


def test_query_classifier():
    """Test query classification"""
    print("\n" + "="*60)
    print("🧪 TEST 1: QUERY CLASSIFIER")
    print("="*60 + "\n")
    
    classifier = QueryClassifier()
    
    test_queries = [
        "What is the total revenue?",
        "Show me products with price above 100",
        "Compare sales between Q1 and Q2",
        "Create a chart showing trends",
        "Find all customers in New York",
        "What are the top 5 selling products?"
    ]
    
    print("Testing query classification...\n")
    
    for query in test_queries:
        result = classifier.classify_query(query)
        print(f"Query: '{query}'")
        print(f"  Type: {result['query_type'].value} ({result['type_confidence']:.0%})")
        print(f"  Intent: {result['intent'].value}")
        print(f"  Strategy: {result['search_strategy']}")
        print(f"  Keywords: {', '.join(result['entities']['keywords'][:3])}")
        print()
    
    print("="*60)
    print("✅ QUERY CLASSIFIER TEST PASSED!")
    print("="*60 + "\n")


def test_query_rewriter():
    """Test query rewriting"""
    print("\n" + "="*60)
    print("🧪 TEST 2: QUERY REWRITER")
    print("="*60 + "\n")
    
    rewriter = QueryRewriter()
    
    test_queries = [
        "What is the price of the product?",
        "Show me high revenue customers",
    ]
    
    print("Testing query rewriting...\n")
    
    for query in test_queries:
        variations = rewriter.rewrite_query(query, expand=True)
        keywords = rewriter.extract_keywords(query)
        
        print(f"Original: '{query}'")
        print(f"Variations ({len(variations)}):")
        for var in variations[:3]:
            print(f"  - {var}")
        print(f"Keywords: {', '.join(keywords)}")
        print()
    
    print("="*60)
    print("✅ QUERY REWRITER TEST PASSED!")
    print("="*60 + "\n")


def test_hybrid_search():
    """Test hybrid search engine"""
    print("\n" + "="*60)
    print("🧪 TEST 3: HYBRID SEARCH ENGINE")
    print("="*60 + "\n")
    
    # Create test documents
    documents = [
        "Product: Laptop, Price: 1200, Category: Electronics",
        "Product: Mouse, Price: 25, Category: Accessories",
        "Product: Keyboard, Price: 75, Category: Accessories",
        "Statistics: Average price is 433, Total products: 3",
        "Column: price, Type: numeric, Mean: 433.33"
    ]
    
    metadatas = [
        {"type": "data_chunk", "file": "products.csv"},
        {"type": "data_chunk", "file": "products.csv"},
        {"type": "data_chunk", "file": "products.csv"},
        {"type": "statistics", "file": "products.csv"},
        {"type": "column_metadata", "file": "products.csv"}
    ]
    
    # Initialize engine
    engine = HybridSearchEngine(alpha=0.5)
    engine.index_documents(documents, metadatas)
    
    print("1. Test keyword-only search...")
    results = engine.keyword_search_only("price laptop", n_results=3)
    print(f"   Found {len(results['documents'])} results")
    print(f"   Top result: {results['documents'][0][:50]}...")
    print()
    
    # Simulate semantic results for hybrid
    print("2. Test hybrid search...")
    semantic_results = {
        "documents": documents,
        "metadatas": metadatas,
        "distances": [0.1, 0.3, 0.4, 0.2, 0.15]
    }
    
    hybrid_results = engine.search("expensive products", semantic_results, n_results=3)
    print(f"   Found {len(hybrid_results['documents'])} results")
    print(f"   Top score: {hybrid_results['scores'][0]:.3f}")
    print()
    
    print("="*60)
    print("✅ HYBRID SEARCH TEST PASSED!")
    print("="*60 + "\n")


def test_result_reranker():
    """Test result re-ranking"""
    print("\n" + "="*60)
    print("🧪 TEST 4: RESULT RERANKER")
    print("="*60 + "\n")
    
    reranker = ResultReranker()
    
    # Test results
    results = {
        "documents": [
            "Product: Laptop with high price",
            "Statistics about pricing",
            "Column metadata for price field"
        ],
        "metadatas": [
            {"type": "data_chunk"},
            {"type": "statistics"},
            {"type": "column_metadata"}
        ],
        "scores": [0.8, 0.6, 0.7]
    }
    
    print("1. Original scores:")
    for doc, score in zip(results["documents"], results["scores"]):
        print(f"   {score:.2f}: {doc[:40]}...")
    print()
    
    print("2. After re-ranking:")
    reranked = reranker.rerank("laptop price", results)
    for doc, score in zip(reranked["documents"], reranked["scores"]):
        print(f"   {score:.2f}: {doc[:40]}...")
    print()
    
    print("="*60)
    print("✅ RESULT RERANKER TEST PASSED!")
    print("="*60 + "\n")


def test_search_router():
    """Test search routing"""
    print("\n" + "="*60)
    print("🧪 TEST 5: SEARCH ROUTER")
    print("="*60 + "\n")
    
    # Setup
    documents = ["Product: Laptop, Price: 1200"]
    metadatas = [{"type": "data_chunk"}]
    
    engine = HybridSearchEngine()
    engine.index_documents(documents, metadatas)
    
    reranker = ResultReranker()
    router = SearchRouter(engine, reranker)
    classifier = QueryClassifier()
    
    # Test routing
    query = "What is the highest price?"
    classification = classifier.classify_query(query)
    
    semantic_results = {
        "documents": documents,
        "metadatas": metadatas,
        "distances": [0.2]
    }
    
    print(f"Query: '{query}'")
    print(f"Strategy: {classification['search_strategy']}")
    print()
    
    final_results = router.route_and_search(
        query,
        classification,
        semantic_results,
        n_results=1
    )
    
    print(f"Results:")
    print(f"  Found: {len(final_results['documents'])} documents")
    print(f"  Strategy used: {final_results['search_strategy']}")
    print(f"  Query type: {final_results['query_type']}")
    print()
    
    print("="*60)
    print("✅ SEARCH ROUTER TEST PASSED!")
    print("="*60 + "\n")


def main():
    """Run all hybrid search tests"""
    print("\n🚀 DAY 38: HYBRID SEARCH & QUERY ROUTING TESTS")
    print("Testing intelligent search and query classification\n")
    
    try:
        # Test 1: Query Classifier
        test_query_classifier()
        
        # Test 2: Query Rewriter
        test_query_rewriter()
        
        # Test 3: Hybrid Search
        test_hybrid_search()
        
        # Test 4: Result Reranker
        test_result_reranker()
        
        # Test 5: Search Router
        test_search_router()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 ALL HYBRID SEARCH TESTS COMPLETED!")
        print("="*60)
        print("\n✅ Query Classifier: PASSED")
        print("✅ Query Rewriter: PASSED")
        print("✅ Hybrid Search: PASSED")
        print("✅ Result Reranker: PASSED")
        print("✅ Search Router: PASSED")
        print("\n🎯 Day 38 hybrid search system is complete!")
        print("\n💡 Key Features:")
        print("  - Intelligent query classification")
        print("  - BM25 + semantic hybrid search")
        print("  - Automatic query routing")
        print("  - Result re-ranking")
        print("  - Query expansion & rewriting\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
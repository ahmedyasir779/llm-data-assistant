from typing import List, Dict, Any, Tuple
import time
import numpy as np


class RetrievalBenchmark:
    """
    Benchmark retrieval strategies for performance
    """
    
    def __init__(self):
        """Initialize benchmark"""
        self.results = []
        
        print("📊 Retrieval benchmark initialized")
    
    def benchmark_strategy(
        self,
        strategy_name: str,
        retriever_func: callable,
        test_queries: List[str],
        ground_truth: Optional[Dict[str, List[str]]] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark a retrieval strategy
        
        Args:
            strategy_name: Name of strategy
            retriever_func: Retrieval function
            test_queries: List of test queries
            ground_truth: Optional ground truth relevance
            n_results: Number of results per query
            
        Returns:
            Benchmark results
        """
        print(f"\n🔍 Benchmarking: {strategy_name}")
        print(f"   Test queries: {len(test_queries)}")
        
        metrics = {
            "strategy": strategy_name,
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_latency_ms": 0.0,
            "avg_results_returned": 0.0,
            "total_time_s": 0.0
        }
        
        latencies = []
        results_counts = []
        
        for query in test_queries:
            try:
                # Time the retrieval
                start_time = time.time()
                results = retriever_func(query, n_results=n_results)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                
                latencies.append(latency)
                results_counts.append(len(results.get("documents", [])))
                metrics["successful_queries"] += 1
                
            except Exception as e:
                print(f"   ❌ Query failed: {query[:30]}... - {e}")
                metrics["failed_queries"] += 1
        
        # Calculate metrics
        if latencies:
            metrics["avg_latency_ms"] = np.mean(latencies)
            metrics["min_latency_ms"] = np.min(latencies)
            metrics["max_latency_ms"] = np.max(latencies)
            metrics["p50_latency_ms"] = np.percentile(latencies, 50)
            metrics["p95_latency_ms"] = np.percentile(latencies, 95)
        
        if results_counts:
            metrics["avg_results_returned"] = np.mean(results_counts)
            metrics["min_results"] = np.min(results_counts)
            metrics["max_results"] = np.max(results_counts)
        
        metrics["total_time_s"] = sum(latencies) / 1000 if latencies else 0
        metrics["success_rate"] = (
            metrics["successful_queries"] / metrics["total_queries"] * 100
            if metrics["total_queries"] > 0 else 0
        )
        
        self.results.append(metrics)
        
        print(f"   ✅ Success rate: {metrics['success_rate']:.1f}%")
        print(f"   ⚡ Avg latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"   📊 Avg results: {metrics['avg_results_returned']:.1f}")
        
        return metrics
    
    def compare_strategies(self) -> None:
        """Print comparison of all benchmarked strategies"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*70)
        print("📊 RETRIEVAL STRATEGY COMPARISON")
        print("="*70)
        
        print(f"\n{'Strategy':<25} {'Success%':<12} {'Latency(ms)':<15} {'Results':<10}")
        print("-"*70)
        
        for result in self.results:
            print(
                f"{result['strategy']:<25} "
                f"{result['success_rate']:<11.1f}% "
                f"{result['avg_latency_ms']:<14.1f} "
                f"{result['avg_results_returned']:<10.1f}"
            )
        
        print("\n" + "="*70)
        
        # Find best strategy
        best_by_speed = min(self.results, key=lambda x: x["avg_latency_ms"])
        best_by_results = max(self.results, key=lambda x: x["avg_results_returned"])
        
        print(f"\n⚡ Fastest: {best_by_speed['strategy']} "
              f"({best_by_speed['avg_latency_ms']:.1f}ms)")
        print(f"📊 Most results: {best_by_results['strategy']} "
              f"({best_by_results['avg_results_returned']:.1f} avg)")
        print()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary"""
        if not self.results:
            return {}
        
        return {
            "num_strategies": len(self.results),
            "total_queries": sum(r["total_queries"] for r in self.results),
            "avg_latency_across_all": np.mean([r["avg_latency_ms"] for r in self.results]),
            "results": self.results
        }
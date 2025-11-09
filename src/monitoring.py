import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque


class PerformanceMonitor:
    """
    Monitor system and application performance
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize performance monitor
        
        Args:
            history_size: Number of metrics to keep in history
        """
        self.history_size = history_size
        self.query_times = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.active_queries = 0
        self.total_queries = 0
        self.failed_queries = 0
        self.start_time = time.time()
        
        print("📊 Performance monitor initialized")
    
    def record_query(self, duration: float, success: bool = True) -> None:
        """
        Record query execution
        
        Args:
            duration: Query duration in seconds
            success: Whether query succeeded
        """
        self.query_times.append(duration)
        self.total_queries += 1
        
        if not success:
            self.failed_queries += 1
    
    def start_query(self) -> float:
        """Start timing a query"""
        self.active_queries += 1
        return time.time()
    
    def end_query(self, start_time: float, success: bool = True) -> float:
        """
        End timing a query
        
        Args:
            start_time: Query start time
            success: Whether query succeeded
            
        Returns:
            Query duration
        """
        duration = time.time() - start_time
        self.active_queries = max(0, self.active_queries - 1)
        self.record_query(duration, success)
        return duration
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files())
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.query_times:
            return {
                "total_queries": self.total_queries,
                "failed_queries": self.failed_queries,
                "active_queries": self.active_queries
            }
        
        import numpy as np
        
        query_times_list = list(self.query_times)
        
        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "success_rate": (
                (self.total_queries - self.failed_queries) / self.total_queries * 100
                if self.total_queries > 0 else 100
            ),
            "active_queries": self.active_queries,
            "avg_query_time_s": np.mean(query_times_list),
            "min_query_time_s": np.min(query_times_list),
            "max_query_time_s": np.max(query_times_list),
            "p50_query_time_s": np.percentile(query_times_list, 50),
            "p95_query_time_s": np.percentile(query_times_list, 95),
            "queries_per_minute": (
                self.total_queries / ((time.time() - self.start_time) / 60)
                if time.time() - self.start_time > 0 else 0
            )
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        perf_stats = self.get_performance_stats()
        sys_metrics = self.get_system_metrics()
        
        # Determine health status
        health = "healthy"
        issues = []
        
        if sys_metrics["memory_percent"] > 80:
            health = "warning"
            issues.append("High memory usage")
        
        if sys_metrics["cpu_percent"] > 80:
            health = "warning"
            issues.append("High CPU usage")
        
        if perf_stats.get("success_rate", 100) < 90:
            health = "degraded"
            issues.append("High failure rate")
        
        if self.active_queries > 10:
            health = "warning"
            issues.append("High concurrent queries")
        
        return {
            "status": health,
            "issues": issues,
            "uptime_seconds": time.time() - self.start_time,
            "performance": perf_stats,
            "system": sys_metrics
        }
    
    def print_summary(self) -> None:
        """Print performance summary"""
        health = self.get_health_status()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\n🏥 Health Status: {health['status'].upper()}")
        if health['issues']:
            print(f"⚠️  Issues: {', '.join(health['issues'])}")
        
        print(f"\n📈 Performance:")
        perf = health['performance']
        print(f"   Total queries: {perf['total_queries']}")
        print(f"   Success rate: {perf.get('success_rate', 100):.1f}%")
        print(f"   Active queries: {perf['active_queries']}")
        
        if 'avg_query_time_s' in perf:
            print(f"   Avg query time: {perf['avg_query_time_s']:.2f}s")
            print(f"   P95 query time: {perf['p95_query_time_s']:.2f}s")
        
        print(f"\n💻 System:")
        sys_m = health['system']
        print(f"   CPU: {sys_m['cpu_percent']:.1f}%")
        print(f"   Memory: {sys_m['memory_mb']:.1f} MB ({sys_m['memory_percent']:.1f}%)")
        print(f"   Threads: {sys_m['threads']}")
        
        print(f"\n⏱️  Uptime: {health['uptime_seconds']/60:.1f} minutes")
        print("="*60 + "\n")


class ApplicationLogger:
    """
    Structured application logging
    """
    
    def __init__(self, name: str = "llm-data-assistant", log_level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        print(f"📝 Logger initialized: {name} (level: {log_level})")
    
    def log_query(self, query: str, duration: float, success: bool = True) -> None:
        """Log query execution"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Query {status}: '{query[:50]}...' (duration: {duration:.2f}s)"
        )
    
    def log_error(self, context: str, error: Exception) -> None:
        """Log error with context"""
        self.logger.error(f"{context}: {type(error).__name__} - {str(error)}")
    
    def log_system_event(self, event: str, details: Dict[str, Any]) -> None:
        """Log system event"""
        self.logger.info(f"System event: {event} - {details}")
    
    def log_performance_warning(self, metric: str, value: float, threshold: float) -> None:
        """Log performance warning"""
        self.logger.warning(
            f"Performance warning: {metric} = {value:.2f} (threshold: {threshold})"
        )


class HealthCheck:
    """
    System health check
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize health check
        
        Args:
            performance_monitor: Performance monitor instance
        """
        self.monitor = performance_monitor
        self.checks = []
        
        print("🏥 Health check initialized")
    
    def add_check(self, name: str, check_func: callable) -> None:
        """
        Add custom health check
        
        Args:
            name: Check name
            check_func: Function that returns (passed: bool, message: str)
        """
        self.checks.append({"name": name, "func": check_func})
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": []
        }
        
        # Run custom checks
        for check in self.checks:
            try:
                passed, message = check["func"]()
                results["checks"].append({
                    "name": check["name"],
                    "passed": passed,
                    "message": message
                })
                
                if not passed:
                    results["overall_status"] = "unhealthy"
                    
            except Exception as e:
                results["checks"].append({
                    "name": check["name"],
                    "passed": False,
                    "message": f"Check failed: {e}"
                })
                results["overall_status"] = "unhealthy"
        
        # Add system health
        health = self.monitor.get_health_status()
        results["system_health"] = health["status"]
        
        if health["status"] != "healthy":
            results["overall_status"] = health["status"]
        
        return results
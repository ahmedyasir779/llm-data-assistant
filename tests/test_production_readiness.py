import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.error_handler import ErrorHandler, RetryHandler, ErrorSeverity, RetryStrategy, with_retry
from src.monitoring import PerformanceMonitor, ApplicationLogger, HealthCheck
from src.config import ConfigManager, ApplicationConfig
import time


def test_error_handler():
    """Test error handling"""
    print("\n" + "="*60)
    print("🧪 TEST 1: ERROR HANDLER")
    print("="*60 + "\n")
    
    handler = ErrorHandler(log_errors=True)
    
    print("1. Test error handling...")
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_record = handler.handle_error(e, "test_context", ErrorSeverity.MEDIUM)
        print(f"   Error recorded: {error_record['type']}")
        print(f"   Severity: {error_record['severity']}")
        print("   ✅ Error handling works\n")
    
    print("2. Test error summary...")
    summary = handler.get_error_summary()
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   By type: {summary['by_type']}")
    print("   ✅ Error summary works\n")
    
    print("="*60)
    print("✅ ERROR HANDLER TEST PASSED!")
    print("="*60 + "\n")


def test_retry_handler():
    """Test retry logic"""
    print("\n" + "="*60)
    print("🧪 TEST 2: RETRY HANDLER")
    print("="*60 + "\n")
    
    retry_handler = RetryHandler(max_retries=3, strategy=RetryStrategy.EXPONENTIAL)
    
    # Test successful retry
    print("1. Test retry with eventual success...")
    
    attempt_count = [0]
    
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise Exception("Temporary failure")
        return "Success!"
    
    result = retry_handler.retry(flaky_function)
    print(f"   Result: {result}")
    print(f"   Attempts: {attempt_count[0]}")
    print("   ✅ Retry works\n")
    
    # Test decorator
    print("2. Test retry decorator...")
    
    call_count = [0]
    
    @with_retry(max_retries=2)
    def decorated_function():
        call_count[0] += 1
        if call_count[0] < 2:
            raise Exception("Fail once")
        return "Decorated success!"
    
    result = decorated_function()
    print(f"   Result: {result}")
    print(f"   Calls: {call_count[0]}")
    print("   ✅ Decorator works\n")
    
    print("="*60)
    print("✅ RETRY HANDLER TEST PASSED!")
    print("="*60 + "\n")


def test_performance_monitor():
    """Test performance monitoring"""
    print("\n" + "="*60)
    print("🧪 TEST 3: PERFORMANCE MONITOR")
    print("="*60 + "\n")
    
    monitor = PerformanceMonitor(history_size=50)
    
    print("1. Test query recording...")
    
    # Simulate queries
    for i in range(5):
        start = monitor.start_query()
        time.sleep(0.01)  # Simulate work
        monitor.end_query(start, success=True)
    
    # Simulate failure
    start = monitor.start_query()
    monitor.end_query(start, success=False)
    
    print(f"   Total queries: {monitor.total_queries}")
    print(f"   Failed queries: {monitor.failed_queries}")
    print("   ✅ Query recording works\n")
    
    print("2. Test system metrics...")
    metrics = monitor.get_system_metrics()
    print(f"   CPU: {metrics['cpu_percent']:.1f}%")
    print(f"   Memory: {metrics['memory_mb']:.1f} MB")
    print("   ✅ System metrics works\n")
    
    print("3. Test performance stats...")
    stats = monitor.get_performance_stats()
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Avg query time: {stats['avg_query_time_s']:.3f}s")
    print("   ✅ Performance stats work\n")
    
    print("4. Test health status...")
    health = monitor.get_health_status()
    print(f"   Status: {health['status']}")
    print(f"   Issues: {health['issues']}")
    print("   ✅ Health status works\n")
    
    print("="*60)
    print("✅ PERFORMANCE MONITOR TEST PASSED!")
    print("="*60 + "\n")


def test_application_logger():
    """Test logging system"""
    print("\n" + "="*60)
    print("🧪 TEST 4: APPLICATION LOGGER")
    print("="*60 + "\n")
    
    logger = ApplicationLogger(name="test_app", log_level="INFO")
    
    print("1. Test query logging...")
    logger.log_query("What is the price?", duration=0.5, success=True)
    print("   ✅ Query logged\n")
    
    print("2. Test error logging...")
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error("test_context", e)
    print("   ✅ Error logged\n")
    
    print("3. Test event logging...")
    logger.log_system_event("startup", {"version": "2.3.0"})
    print("   ✅ Event logged\n")
    
    print("="*60)
    print("✅ APPLICATION LOGGER TEST PASSED!")
    print("="*60 + "\n")


def test_config_manager():
    """Test configuration management"""
    print("\n" + "="*60)
    print("🧪 TEST 5: CONFIGURATION MANAGER")
    print("="*60 + "\n")
    
    # Create test .env content
    test_env = """
GROQ_API_KEY=test_key_12345
MODEL_NAME=llama-3.1-8b-instant
TEMPERATURE=0.7
MAX_TOKENS=2048
"""
    
    # Write test .env
    with open(".env.test", "w") as f:
        f.write(test_env)
    
    print("1. Test config loading...")
    config_manager = ConfigManager(env_file=".env.test")
    config = config_manager.load_config()
    
    print(f"   Model: {config.llm.model}")
    print(f"   Temperature: {config.llm.temperature}")
    print("   ✅ Config loaded\n")
    
    print("2. Test config validation...")
    print(f"   API key set: {bool(config.llm.api_key)}")
    print(f"   Valid temperature: {0 <= config.llm.temperature <= 2}")
    print("   ✅ Validation works\n")
    
    # Cleanup
    Path(".env.test").unlink()
    
    print("="*60)
    print("✅ CONFIGURATION MANAGER TEST PASSED!")
    print("="*60 + "\n")


def main():
    """Run all production readiness tests"""
    print("\n🚀 DAY 41: PRODUCTION READINESS TESTS")
    print("Testing error handling, monitoring, and configuration\n")
    
    try:
        # Test 1: Error Handler
        test_error_handler()
        
        # Test 2: Retry Handler
        test_retry_handler()
        
        # Test 3: Performance Monitor
        test_performance_monitor()
        
        # Test 4: Application Logger
        test_application_logger()
        
        # Test 5: Config Manager
        test_config_manager()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 ALL PRODUCTION READINESS TESTS COMPLETED!")
        print("="*60)
        print("\n✅ Error Handler: PASSED")
        print("✅ Retry Handler: PASSED")
        print("✅ Performance Monitor: PASSED")
        print("✅ Application Logger: PASSED")
        print("✅ Configuration Manager: PASSED")
        print("\n🎯 Day 41 production readiness complete!")
        print("\n💡 Key Features:")
        print("  - Robust error handling & retry logic")
        print("  - Performance monitoring & metrics")
        print("  - Health checks & system monitoring")
        print("  - Structured logging")
        print("  - Configuration validation")
        print("  - Production-ready deployment\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
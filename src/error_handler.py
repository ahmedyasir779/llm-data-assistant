import time
import logging
from typing import Callable, Any, Optional, Dict, List, Tuple
from functools import wraps
from enum import Enum
import traceback


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Recoverable, minor issues
    MEDIUM = "medium"     # Needs attention but not critical
    HIGH = "high"         # Critical, needs immediate attention
    CRITICAL = "critical" # System failure


class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL = "exponential"  # 1s, 2s, 4s, 8s...
    LINEAR = "linear"            # 1s, 2s, 3s, 4s...
    CONSTANT = "constant"        # 1s, 1s, 1s, 1s...


class ErrorHandler:
    """
    Centralized error handling system
    """
    
    def __init__(self, log_errors: bool = True):
        """
        Initialize error handler
        
        Args:
            log_errors: Whether to log errors
        """
        self.log_errors = log_errors
        self.error_counts = {}
        self.error_history = []
        
        # Setup logging
        if log_errors:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        
        print("🛡️ Error handler initialized")
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Dict[str, Any]:
        """
        Handle and log error
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            severity: Error severity level
            
        Returns:
            Error information dictionary
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Track error counts
        error_key = f"{context}:{error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create error record
        error_record = {
            "type": error_type,
            "message": error_msg,
            "context": context,
            "severity": severity.value,
            "count": self.error_counts[error_key],
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        
        # Add to history
        self.error_history.append(error_record)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Log based on severity
        if self.log_errors:
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"{context}: {error_type} - {error_msg}")
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(f"{context}: {error_type} - {error_msg}")
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"{context}: {error_type} - {error_msg}")
            else:
                self.logger.info(f"{context}: {error_type} - {error_msg}")
        
        return error_record
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error statistics summary"""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count by type
        by_type = {}
        by_severity = {}
        
        for error in self.error_history:
            error_type = error["type"]
            severity = error["severity"]
            
            by_type[error_type] = by_type.get(error_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "most_common": max(by_type.items(), key=lambda x: x[1]) if by_type else None,
            "recent_errors": self.error_history[-5:]
        }
    
    def clear_history(self) -> None:
        """Clear error history"""
        self.error_history = []
        self.error_counts = {}
        print("✅ Error history cleared")


class RetryHandler:
    """
    Retry logic for transient failures
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            strategy: Retry strategy to use
            base_delay: Base delay in seconds
        """
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        
        print(f"🔄 Retry handler initialized")
        print(f"   Max retries: {max_retries}")
        print(f"   Strategy: {strategy.value}")
    
    def retry(
        self,
        func: Callable,
        *args,
        retry_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            retry_exceptions: Tuple of exceptions to retry on
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except retry_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    print(f"   ⚠️ Attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ All {self.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on strategy"""
        if self.strategy == RetryStrategy.EXPONENTIAL:
            return self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            return self.base_delay * (attempt + 1)
        else:  # CONSTANT
            return self.base_delay


def with_retry(
    max_retries: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_exceptions: tuple = (Exception,)
):
    """
    Decorator for automatic retry
    
    Args:
        max_retries: Maximum retry attempts
        strategy: Retry strategy
        retry_exceptions: Exceptions to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(max_retries, strategy)
            return retry_handler.retry(
                func,
                *args,
                retry_exceptions=retry_exceptions,
                **kwargs
            )
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_handler: Optional[ErrorHandler] = None,
    context: str = "unknown",
    **kwargs
) -> Any:
    """
    Safely execute function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default value if error occurs
        error_handler: Error handler instance
        context: Context description
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e, context)
        else:
            print(f"❌ Error in {context}: {e}")
        
        return default_return
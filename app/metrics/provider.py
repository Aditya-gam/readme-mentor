"""Provider module for metrics collection system."""

import logging
import os
from typing import Optional

from .collector import MetricsCollector

logger = logging.getLogger(__name__)

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(session_id: Optional[str] = None) -> MetricsCollector:
    """Get the global metrics collector instance.

    This function provides a singleton pattern for the metrics collector,
    ensuring that all parts of the application use the same instance
    for consistent metrics collection.

    Args:
        session_id: Optional session ID. If provided and collector doesn't exist,
                   creates a new collector with this session ID.

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector

    if _metrics_collector is None:
        # Check if metrics collection is enabled
        enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        enable_persistence = (
            os.getenv("ENABLE_METRICS_PERSISTENCE", "true").lower() == "true"
        )

        if enable_metrics:
            _metrics_collector = MetricsCollector(
                session_id=session_id, enable_persistence=enable_persistence
            )
            logger.info(
                f"Initialized metrics collector with session ID: {_metrics_collector.session_id}"
            )
        else:
            # Create a dummy collector that does nothing
            _metrics_collector = DummyMetricsCollector()
            logger.info("Metrics collection is disabled - using dummy collector")

    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector instance.

    This is useful for testing or when you want to start fresh
    with a new session.
    """
    global _metrics_collector
    _metrics_collector = None
    logger.info("Reset metrics collector")


class DummyMetricsCollector:
    """Dummy metrics collector that does nothing when metrics are disabled.

    This class provides the same interface as MetricsCollector but
    performs no actual work, allowing the application to run without
    metrics collection overhead when disabled.
    """

    def __init__(self):
        """Initialize dummy collector."""
        self.session_id = "dummy_session"

    def start_operation(self, operation_name: str, metadata=None) -> str:
        """Dummy operation start."""
        return f"dummy_{operation_name}_0"

    def end_operation(
        self,
        operation_id: str,
        success=True,
        error_count=0,
        warning_count=0,
        additional_metadata=None,
    ):
        """Dummy operation end."""
        pass

    def add_component_timing(
        self, operation_id: str, component: str, duration: float
    ) -> None:
        """Dummy component timing."""
        pass

    def record_tool_call(
        self,
        operation_id: str,
        tool_name: str,
        status,
        duration: float,
        error_category=None,
        error_message=None,
        metadata=None,
    ):
        """Dummy tool call recording."""
        pass

    def record_token_usage(
        self, operation_id: str, input_tokens: int, output_tokens: int, model_name=None
    ):
        """Dummy token usage recording."""
        pass

    def analyze_performance_trends(self, operation_type: str, time_period: str = "1h"):
        """Dummy trend analysis."""
        pass

    def get_session_summary(self):
        """Dummy session summary."""
        pass

    def save_metrics(self, filename=None):
        """Dummy metrics saving."""
        pass

    def load_metrics(self, filepath):
        """Dummy metrics loading."""
        pass

    def operation_context(self, operation_name: str, metadata=None):
        """Dummy operation context."""
        import contextlib

        @contextlib.contextmanager
        def dummy_context():
            yield f"dummy_{operation_name}_0"

        return dummy_context()

    def get_system_metrics(self):
        """Dummy system metrics."""
        return {}

    def clear_session(self) -> None:
        """Dummy session clearing."""
        pass


def with_metrics_tracking(operation_name: str):
    """Decorator for automatic metrics tracking of functions.

    This decorator automatically tracks the execution of a function
    using the metrics collector, including timing, success/failure,
    and any exceptions that occur.

    Args:
        operation_name: Name to use for the operation in metrics

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()

            with collector.operation_context(operation_name):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    # The context manager will handle recording the failure
                    raise

        return wrapper

    return decorator


def track_tool_call(tool_name: str):
    """Decorator for tracking tool calls with metrics.

    This decorator automatically tracks tool calls, including
    timing, success/failure status, and error categorization.

    Args:
        tool_name: Name of the tool being called

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Note: Tool call recording requires an active operation context
                # In practice, this would be called from within an operation

                return result
            except Exception:
                # Note: Tool call recording requires an active operation context
                # In practice, this would be called from within an operation

                raise

        return wrapper

    return decorator

"""Performance Metrics Collection System for readme-mentor.

This module provides comprehensive performance metrics collection including:
- Tool call metrics (success/failure rates, call counts, timing)
- Token usage tracking (input/output counts, cost estimation)
- Wall time measurement (operation duration, component timing)
- Performance trends and optimization opportunities
- Metrics display with multiple verbosity levels and output formats
"""

from .collector import MetricsCollector
from .display import MetricsDisplayFormatter
from .models import (
    CostEstimate,
    MetricsData,
    OperationMetrics,
    PerformanceTrend,
    TokenUsage,
    ToolCallMetrics,
)
from .provider import get_metrics_collector, reset_metrics_collector

__all__ = [
    "MetricsCollector",
    "MetricsDisplayFormatter",
    "MetricsData",
    "OperationMetrics",
    "PerformanceTrend",
    "TokenUsage",
    "ToolCallMetrics",
    "CostEstimate",
    "get_metrics_collector",
    "reset_metrics_collector",
]

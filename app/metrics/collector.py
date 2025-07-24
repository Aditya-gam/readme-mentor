"""Centralized metrics collector for performance monitoring."""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

from .models import (
    CostEstimate,
    ErrorCategory,
    MetricsData,
    OperationMetrics,
    PerformanceTrend,
    TokenUsage,
    ToolCallMetrics,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Centralized collector for performance metrics."""

    def __init__(
        self, session_id: Optional[str] = None, enable_persistence: bool = True
    ):
        """Initialize the metrics collector.

        Args:
            session_id: Unique session identifier. If None, one will be generated.
            enable_persistence: Whether to persist metrics to disk.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.enable_persistence = enable_persistence

        # Current session data
        self.current_session = MetricsData(
            session_id=self.session_id,
            start_time=datetime.now(),
        )

        # Active operations tracking
        self.active_operations: Dict[str, OperationMetrics] = {}

        # Historical data for trend analysis
        self.historical_operations: List[OperationMetrics] = []

        # Model pricing information for cost estimation
        self.model_pricing = {
            # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},  # per 1K tokens
            "llama3:8b": {"input": 0.0, "output": 0.0},  # local model, no cost
            # local model, no cost
            "llama3.1:8b": {"input": 0.0, "output": 0.0},
        }

        # Performance thresholds for bottleneck detection
        self.performance_thresholds = {
            "qa_response": 3.0,  # seconds
            "vector_search": 0.5,  # seconds
            "llm_inference": 2.0,  # seconds
            "embedding_generation": 1.0,  # seconds
        }

        # Setup persistence directory
        if self.enable_persistence:
            self.persistence_dir = Path("data/metrics")
            self.persistence_dir.mkdir(parents=True, exist_ok=True)

    def start_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking an operation.

        Args:
            operation_name: Name of the operation to track
            metadata: Additional metadata for the operation

        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"

        operation = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {},
        )

        self.active_operations[operation_id] = operation
        logger.debug(f"Started operation tracking: {operation_id}")

        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_count: int = 0,
        warning_count: int = 0,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationMetrics:
        """End tracking an operation.

        Args:
            operation_id: Operation ID returned from start_operation
            success: Whether the operation was successful
            error_count: Number of errors encountered
            warning_count: Number of warnings encountered
            additional_metadata: Additional metadata to add

        Returns:
            Completed operation metrics

        Raises:
            KeyError: If operation_id is not found
        """
        if operation_id not in self.active_operations:
            raise KeyError(f"Operation ID not found: {operation_id}")

        operation = self.active_operations[operation_id]
        operation.end_time = time.time()
        operation.success = success
        operation.error_count = error_count
        operation.warning_count = warning_count

        # Calculate total duration
        if operation.end_time and operation.start_time:
            operation.total_duration = operation.end_time - operation.start_time

        if additional_metadata:
            operation.metadata.update(additional_metadata)

        # Move to completed operations
        self.current_session.operations.append(operation)
        self.historical_operations.append(operation)
        del self.active_operations[operation_id]

        logger.debug(
            f"Completed operation: {operation_id} (duration: {operation.total_duration:.3f}s)"
        )

        return operation

    def add_component_timing(
        self, operation_id: str, component: str, duration: float
    ) -> None:
        """Add component timing to an active operation.

        Args:
            operation_id: Operation ID
            component: Component name
            duration: Duration in seconds

        Raises:
            KeyError: If operation_id is not found
        """
        if operation_id not in self.active_operations:
            raise KeyError(f"Operation ID not found: {operation_id}")

        self.active_operations[operation_id].component_timing[component] = duration
        logger.debug(f"Added component timing: {component} = {duration:.3f}s")

    def record_tool_call(
        self,
        operation_id: str,
        tool_name: str,
        status: ToolCallStatus,
        duration: float,
        error_category: Optional[ErrorCategory] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolCallMetrics:
        """Record a tool call for an operation.

        Args:
            operation_id: Operation ID
            tool_name: Name of the tool
            status: Status of the tool call
            duration: Duration of the tool call
            error_category: Error category if failed
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            Tool call metrics

        Raises:
            KeyError: If operation_id is not found
        """
        if operation_id not in self.active_operations:
            raise KeyError(f"Operation ID not found: {operation_id}")

        tool_call = ToolCallMetrics(
            tool_name=tool_name,
            status=status,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            error_category=error_category,
            error_message=error_message,
            metadata=metadata or {},
        )

        self.active_operations[operation_id].tool_calls.append(tool_call)
        logger.debug(f"Recorded tool call: {tool_name} ({status}) in {duration:.3f}s")

        return tool_call

    def record_token_usage(
        self,
        operation_id: str,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None,
    ) -> TokenUsage:
        """Record token usage for an operation.

        Args:
            operation_id: Operation ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model used for tokenization

        Returns:
            Token usage metrics

        Raises:
            KeyError: If operation_id is not found
        """
        if operation_id not in self.active_operations:
            raise KeyError(f"Operation ID not found: {operation_id}")

        token_usage = TokenUsage(
            operation=self.active_operations[operation_id].operation_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
        )

        self.active_operations[operation_id].token_usage = token_usage

        # Calculate cost estimate
        if model_name and model_name in self.model_pricing:
            cost_estimate = self._calculate_cost(token_usage, model_name)
            self.active_operations[operation_id].cost_estimate = cost_estimate

        logger.debug(
            f"Recorded token usage: {input_tokens} input, {output_tokens} output"
        )

        return token_usage

    def _calculate_cost(self, token_usage: TokenUsage, model_name: str) -> CostEstimate:
        """Calculate cost estimate for token usage.

        Args:
            token_usage: Token usage metrics
            model_name: Model name for pricing

        Returns:
            Cost estimate
        """
        pricing = self.model_pricing.get(model_name, {"input": 0.0, "output": 0.0})

        input_cost = (token_usage.input_tokens / 1000) * pricing["input"]
        output_cost = (token_usage.output_tokens / 1000) * pricing["output"]

        return CostEstimate(
            operation=token_usage.operation,
            input_cost=input_cost,
            output_cost=output_cost,
            model_name=model_name,
        )

    def analyze_performance_trends(
        self, operation_type: str, time_period: str = "1h"
    ) -> PerformanceTrend:
        """Analyze performance trends for an operation type.

        Args:
            operation_type: Type of operation to analyze
            time_period: Time period for analysis (e.g., "1h", "24h", "7d")

        Returns:
            Performance trend analysis
        """
        # Filter operations by type and time period
        cutoff_time = self._get_cutoff_time(time_period)
        relevant_operations = [
            op
            for op in self.historical_operations
            if op.operation_name == operation_type and op.start_time >= cutoff_time
        ]

        if not relevant_operations:
            return PerformanceTrend(
                operation_type=operation_type,
                time_period=time_period,
                data_points=0,
                avg_duration=0.0,
                min_duration=0.0,
                max_duration=0.0,
                std_deviation=0.0,
                success_rate=0.0,
                error_rate=0.0,
                avg_cost=0.0,
                total_cost=0.0,
                avg_tokens=0.0,
                total_tokens=0,
                trend_direction="stable",
                trend_strength=0.0,
            )

        # Calculate basic metrics
        durations = [
            op.total_duration for op in relevant_operations if op.total_duration
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        # Calculate standard deviation
        if len(durations) > 1:
            mean = avg_duration
            variance = sum((x - mean) ** 2 for x in durations) / (len(durations) - 1)
            std_deviation = variance**0.5
        else:
            std_deviation = 0.0

        # Calculate success rates
        successful = sum(1 for op in relevant_operations if op.success)
        success_rate = (successful / len(relevant_operations)) * 100
        error_rate = 100 - success_rate

        # Calculate cost metrics
        cost_estimates = [
            op.cost_estimate for op in relevant_operations if op.cost_estimate
        ]
        total_cost = sum(est.total_cost for est in cost_estimates)
        avg_cost = total_cost / len(cost_estimates) if cost_estimates else 0.0

        # Calculate token metrics
        token_usages = [op.token_usage for op in relevant_operations if op.token_usage]
        total_tokens = sum(usage.total_tokens for usage in token_usages)
        avg_tokens = total_tokens / len(token_usages) if token_usages else 0.0

        # Analyze trend direction
        trend_direction, trend_strength = self._analyze_trend_direction(
            relevant_operations
        )

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(relevant_operations)

        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            relevant_operations, bottlenecks
        )

        return PerformanceTrend(
            operation_type=operation_type,
            time_period=time_period,
            data_points=len(relevant_operations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            std_deviation=std_deviation,
            success_rate=success_rate,
            error_rate=error_rate,
            avg_cost=avg_cost,
            total_cost=total_cost,
            avg_tokens=avg_tokens,
            total_tokens=total_tokens,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            bottlenecks=bottlenecks,
            optimization_suggestions=optimization_suggestions,
        )

    def _get_cutoff_time(self, time_period: str) -> float:
        """Get cutoff time for trend analysis.

        Args:
            time_period: Time period string (e.g., "1h", "24h", "7d")

        Returns:
            Cutoff time in seconds since epoch
        """
        now = time.time()

        if time_period.endswith("h"):
            hours = int(time_period[:-1])
            return now - (hours * 3600)
        elif time_period.endswith("d"):
            days = int(time_period[:-1])
            return now - (days * 86400)
        elif time_period.endswith("m"):
            minutes = int(time_period[:-1])
            return now - (minutes * 60)
        else:
            # Default to 1 hour
            return now - 3600

    def _analyze_trend_direction(
        self, operations: List[OperationMetrics]
    ) -> tuple[str, float]:
        """Analyze the trend direction of operations.

        Args:
            operations: List of operations to analyze

        Returns:
            Tuple of (trend_direction, trend_strength)
        """
        if len(operations) < 2:
            return "stable", 0.0

        # Sort by start time
        sorted_ops = sorted(operations, key=lambda x: x.start_time)

        # Calculate trend using linear regression
        durations = [op.total_duration for op in sorted_ops if op.total_duration]
        if len(durations) < 2:
            return "stable", 0.0

        # Simple linear regression
        n = len(durations)
        x_values = list(range(n))
        y_values = durations

        # Calculate slope
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
        sum_x2 = sum(x * x for x in x_values)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Normalize slope to get trend strength (-1 to 1)
        max_duration = max(durations)
        trend_strength = max(
            -1.0, min(1.0, slope / max_duration if max_duration > 0 else 0.0)
        )

        # Determine trend direction
        if trend_strength > 0.1:
            trend_direction = "declining"  # Duration increasing
        elif trend_strength < -0.1:
            trend_direction = "improving"  # Duration decreasing
        else:
            trend_direction = "stable"

        return trend_direction, trend_strength

    def _identify_bottlenecks(self, operations: List[OperationMetrics]) -> List[str]:
        """Identify performance bottlenecks in operations.

        Args:
            operations: List of operations to analyze

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        # Check for slow operations
        slow_operations = [
            op
            for op in operations
            if op.total_duration
            and op.total_duration
            > self.performance_thresholds.get(op.operation_name, 5.0)
        ]

        if slow_operations:
            bottlenecks.append(
                f"{len(slow_operations)} operations exceeded performance thresholds"
            )

        # Check for high error rates
        error_operations = [op for op in operations if not op.success]
        if error_operations:
            error_rate = (len(error_operations) / len(operations)) * 100
            if error_rate > 10:  # More than 10% error rate
                bottlenecks.append(f"High error rate: {error_rate:.1f}%")

        # Check for expensive operations
        expensive_operations = [
            op
            for op in operations
            if op.cost_estimate and op.cost_estimate.total_cost > 0.1  # More than $0.10
        ]
        if expensive_operations:
            bottlenecks.append(
                f"{len(expensive_operations)} operations exceeded cost threshold"
            )

        # Check component timing
        for operation in operations:
            for component, duration in operation.component_timing.items():
                threshold = self.performance_thresholds.get(component, 1.0)
                if duration > threshold:
                    bottlenecks.append(
                        f"Slow {component}: {duration:.2f}s > {threshold}s"
                    )

        return list(set(bottlenecks))  # Remove duplicates

    def _generate_optimization_suggestions(
        self, operations: List[OperationMetrics], bottlenecks: List[str]
    ) -> List[str]:
        """Generate optimization suggestions based on bottlenecks.

        Args:
            operations: List of operations
            bottlenecks: Identified bottlenecks

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Check for slow vector search
        slow_vector_search = any(
            "vector_search" in bottleneck for bottleneck in bottlenecks
        )
        if slow_vector_search:
            suggestions.append(
                "Consider optimizing vector search with better indexing or chunking"
            )

        # Check for slow LLM inference
        slow_llm = any("llm_inference" in bottleneck for bottleneck in bottlenecks)
        if slow_llm:
            suggestions.append("Consider using a faster model or optimizing prompts")

        # Check for high costs
        high_costs = any("cost threshold" in bottleneck for bottleneck in bottlenecks)
        if high_costs:
            suggestions.append(
                "Consider using a cheaper model or optimizing token usage"
            )

        # Check for high error rates
        high_errors = any("error rate" in bottleneck for bottleneck in bottlenecks)
        if high_errors:
            suggestions.append("Review error handling and retry logic")

        # General suggestions
        if len(operations) > 10:
            suggestions.append(
                "Consider implementing caching for frequently accessed data"
            )

        return suggestions

    def get_session_summary(self) -> MetricsData:
        """Get a summary of the current session.

        Returns:
            Session metrics summary
        """
        # Update the current session with latest data
        self.current_session.end_time = datetime.now()
        return self.current_session

    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """Save metrics to disk.

        Args:
            filename: Optional filename. If None, auto-generate based on session ID.

        Returns:
            Path to saved file
        """
        if not self.enable_persistence:
            raise RuntimeError("Persistence is disabled")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{self.session_id}_{timestamp}.json"

        filepath = self.persistence_dir / filename

        # Get session summary
        session_data = self.get_session_summary()

        # Convert to dict for JSON serialization
        data = session_data.model_dump()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved metrics to {filepath}")
        return filepath

    def load_metrics(self, filepath: Union[str, Path]) -> MetricsData:
        """Load metrics from disk.

        Args:
            filepath: Path to metrics file

        Returns:
            Loaded metrics data
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return MetricsData(**data)

    @contextmanager
    def operation_context(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracking operations.

        Args:
            operation_name: Name of the operation
            metadata: Additional metadata

        Yields:
            Operation ID for use within the context
        """
        operation_id = self.start_operation(operation_name, metadata)
        try:
            yield operation_id
            self.end_operation(operation_id, success=True)
        except Exception:
            self.end_operation(operation_id, success=False, error_count=1)
            raise

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.

        Returns:
            Dictionary of system metrics
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage("/").percent,
        }

    def clear_session(self) -> None:
        """Clear the current session data."""
        self.current_session = MetricsData(
            session_id=self.session_id,
            start_time=datetime.now(),
        )
        self.active_operations.clear()
        logger.info("Cleared current session data")

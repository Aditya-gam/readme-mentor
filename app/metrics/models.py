"""Data models for performance metrics collection system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Types of metrics that can be collected."""

    TOOL_CALL = "tool_call"
    TOKEN_USAGE = "token_usage"
    WALL_TIME = "wall_time"
    PERFORMANCE_TREND = "performance_trend"
    ERROR_RATE = "error_rate"
    COST_ESTIMATE = "cost_estimate"


class ToolCallStatus(str, Enum):
    """Status of a tool call."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class ErrorCategory(str, Enum):
    """Categories of errors for tool calls."""

    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ToolCallMetrics(BaseModel):
    """Metrics for a single tool call."""

    tool_name: str = Field(..., description="Name of the tool being called")
    status: ToolCallStatus = Field(..., description="Status of the tool call")
    start_time: float = Field(..., description="Start time in seconds since epoch")
    end_time: Optional[float] = Field(
        None, description="End time in seconds since epoch"
    )
    duration: Optional[float] = Field(None, description="Duration in seconds")
    error_category: Optional[ErrorCategory] = Field(
        None, description="Error category if failed"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __post_init__(self) -> None:
        """Calculate duration if end_time is provided."""
        if self.end_time and not self.duration:
            self.duration = self.end_time - self.start_time


class TokenUsage(BaseModel):
    """Token usage metrics for an operation."""

    operation: str = Field(..., description="Operation name")
    input_tokens: int = Field(0, description="Number of input tokens")
    output_tokens: int = Field(0, description="Number of output tokens")
    total_tokens: int = Field(0, description="Total tokens used")
    model_name: Optional[str] = Field(None, description="Model used for tokenization")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of usage"
    )

    def __post_init__(self) -> None:
        """Calculate total tokens if not provided."""
        if not self.total_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens


class CostEstimate(BaseModel):
    """Cost estimation for token usage."""

    operation: str = Field(..., description="Operation name")
    input_cost: float = Field(0.0, description="Cost for input tokens")
    output_cost: float = Field(0.0, description="Cost for output tokens")
    total_cost: float = Field(0.0, description="Total cost")
    currency: str = Field("USD", description="Currency for cost")
    model_name: str = Field(..., description="Model used for cost calculation")
    pricing_tier: Optional[str] = Field(None, description="Pricing tier used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of estimate"
    )

    def __post_init__(self) -> None:
        """Calculate total cost if not provided."""
        if not self.total_cost:
            self.total_cost = self.input_cost + self.output_cost


class OperationMetrics(BaseModel):
    """Metrics for a complete operation."""

    operation_name: str = Field(..., description="Name of the operation")
    start_time: float = Field(..., description="Start time in seconds since epoch")
    end_time: Optional[float] = Field(
        None, description="End time in seconds since epoch"
    )
    total_duration: Optional[float] = Field(
        None, description="Total duration in seconds"
    )

    # Component timing breakdown
    component_timing: Dict[str, float] = Field(
        default_factory=dict, description="Timing breakdown by component"
    )

    # Tool call metrics
    tool_calls: List[ToolCallMetrics] = Field(
        default_factory=list, description="Tool calls made during operation"
    )

    # Token usage
    token_usage: Optional[TokenUsage] = Field(
        None, description="Token usage for operation"
    )
    cost_estimate: Optional[CostEstimate] = Field(
        None, description="Cost estimate for operation"
    )

    # Performance indicators
    success: bool = Field(True, description="Whether operation was successful")
    error_count: int = Field(0, description="Number of errors encountered")
    warning_count: int = Field(0, description="Number of warnings encountered")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __post_init__(self) -> None:
        """Calculate total duration and error counts."""
        if self.end_time and not self.total_duration:
            self.total_duration = self.end_time - self.start_time

        # Count errors from tool calls
        if not self.error_count:
            self.error_count = sum(
                1
                for call in self.tool_calls
                if call.status
                in [
                    ToolCallStatus.FAILURE,
                    ToolCallStatus.ERROR,
                    ToolCallStatus.TIMEOUT,
                ]
            )


class PerformanceTrend(BaseModel):
    """Performance trend analysis for an operation type."""

    operation_type: str = Field(..., description="Type of operation")
    time_period: str = Field(..., description="Time period for trend analysis")
    data_points: int = Field(..., description="Number of data points analyzed")

    # Trend metrics
    avg_duration: float = Field(..., description="Average duration")
    min_duration: float = Field(..., description="Minimum duration")
    max_duration: float = Field(..., description="Maximum duration")
    std_deviation: float = Field(..., description="Standard deviation")

    # Success metrics
    success_rate: float = Field(..., description="Success rate percentage")
    error_rate: float = Field(..., description="Error rate percentage")

    # Cost metrics
    avg_cost: float = Field(..., description="Average cost per operation")
    total_cost: float = Field(..., description="Total cost for period")

    # Token metrics
    avg_tokens: float = Field(..., description="Average tokens per operation")
    total_tokens: int = Field(..., description="Total tokens for period")

    # Trend indicators
    trend_direction: str = Field(
        ..., description="Trend direction (improving/declining/stable)"
    )
    trend_strength: float = Field(..., description="Trend strength (-1 to 1)")

    # Optimization opportunities
    bottlenecks: List[str] = Field(
        default_factory=list, description="Identified bottlenecks"
    )
    optimization_suggestions: List[str] = Field(
        default_factory=list, description="Optimization suggestions"
    )

    # Metadata
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="When analysis was performed"
    )


class MetricsData(BaseModel):
    """Complete metrics data for a session or time period."""

    session_id: str = Field(..., description="Unique session identifier")
    start_time: datetime = Field(..., description="Session start time")
    end_time: Optional[datetime] = Field(None, description="Session end time")

    # Operation metrics
    operations: List[OperationMetrics] = Field(
        default_factory=list, description="All operations in session"
    )

    # Aggregated metrics
    total_operations: int = Field(0, description="Total number of operations")
    successful_operations: int = Field(0, description="Number of successful operations")
    failed_operations: int = Field(0, description="Number of failed operations")

    # Timing metrics
    total_duration: float = Field(0.0, description="Total session duration")
    avg_operation_duration: float = Field(0.0, description="Average operation duration")

    # Token metrics
    total_input_tokens: int = Field(0, description="Total input tokens used")
    total_output_tokens: int = Field(0, description="Total output tokens used")
    total_tokens: int = Field(0, description="Total tokens used")

    # Cost metrics
    total_cost: float = Field(0.0, description="Total cost for session")
    avg_cost_per_operation: float = Field(0.0, description="Average cost per operation")

    # Tool call metrics
    total_tool_calls: int = Field(0, description="Total tool calls made")
    successful_tool_calls: int = Field(0, description="Successful tool calls")
    failed_tool_calls: int = Field(0, description="Failed tool calls")

    # Performance trends
    trends: List[PerformanceTrend] = Field(
        default_factory=list, description="Performance trends analysis"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __post_init__(self) -> None:
        """Calculate aggregated metrics from operations."""
        if not self.operations:
            return

        # Calculate totals
        self.total_operations = len(self.operations)
        self.successful_operations = sum(1 for op in self.operations if op.success)
        self.failed_operations = self.total_operations - self.successful_operations

        # Calculate timing
        durations = [op.total_duration for op in self.operations if op.total_duration]
        if durations:
            self.total_duration = sum(durations)
            self.avg_operation_duration = self.total_duration / len(durations)

        # Calculate token usage
        token_usages = [op.token_usage for op in self.operations if op.token_usage]
        if token_usages:
            self.total_input_tokens = sum(usage.input_tokens for usage in token_usages)
            self.total_output_tokens = sum(
                usage.output_tokens for usage in token_usages
            )
            self.total_tokens = sum(usage.total_tokens for usage in token_usages)

        # Calculate costs
        cost_estimates = [
            op.cost_estimate for op in self.operations if op.cost_estimate
        ]
        if cost_estimates:
            self.total_cost = sum(estimate.total_cost for estimate in cost_estimates)
            self.avg_cost_per_operation = self.total_cost / len(cost_estimates)

        # Calculate tool call metrics
        all_tool_calls = []
        for op in self.operations:
            all_tool_calls.extend(op.tool_calls)

        self.total_tool_calls = len(all_tool_calls)
        self.successful_tool_calls = sum(
            1 for call in all_tool_calls if call.status == ToolCallStatus.SUCCESS
        )
        self.failed_tool_calls = self.total_tool_calls - self.successful_tool_calls

#!/usr/bin/env python3
"""
Demo script for the Performance Metrics Integration System (Phase 5).

This script demonstrates all the features of the comprehensive metrics collection system:
- Tool call metrics (success/failure rates, call counts, timing)
- Token usage tracking (input/output counts, cost estimation)
- Wall time measurement (operation duration, component timing)
- Performance trends and optimization opportunities

Usage:
    python demo_metrics_system.py
"""

import sys
import time
from pathlib import Path

from app.metrics import get_metrics_collector, reset_metrics_collector
from app.metrics.models import ErrorCategory, ToolCallStatus
from app.metrics.provider import with_metrics_tracking

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def demo_basic_metrics_collection():
    """Demonstrate basic metrics collection functionality."""
    print("🔧 Demo 1: Basic Metrics Collection")
    print("=" * 50)

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    print(f"Session ID: {collector.session_id}")

    # Start an operation
    operation_id = collector.start_operation(
        "demo_operation", metadata={"demo_type": "basic", "user": "demo_user"}
    )
    print(f"Started operation: {operation_id}")

    # Simulate some work
    time.sleep(0.1)

    # Add component timing
    collector.add_component_timing(operation_id, "data_processing", 0.05)
    collector.add_component_timing(operation_id, "api_call", 0.03)
    collector.add_component_timing(operation_id, "formatting", 0.02)

    # Record some tool calls
    collector.record_tool_call(
        operation_id=operation_id,
        tool_name="database_query",
        status=ToolCallStatus.SUCCESS,
        duration=0.05,
        metadata={"query_type": "select", "rows_returned": 100},
    )

    collector.record_tool_call(
        operation_id=operation_id,
        tool_name="external_api",
        status=ToolCallStatus.SUCCESS,
        duration=0.03,
        metadata={"endpoint": "/api/data", "status_code": 200},
    )

    # Record token usage
    collector.record_token_usage(
        operation_id=operation_id,
        input_tokens=150,
        output_tokens=75,
        model_name="gpt-3.5-turbo",
    )

    # End the operation
    operation = collector.end_operation(operation_id, success=True)

    print(f"Operation completed in {operation.total_duration:.3f}s")
    print(f"Component timing: {operation.component_timing}")
    print(f"Tool calls: {len(operation.tool_calls)}")
    print(f"Token usage: {operation.token_usage}")
    print(
        f"Cost estimate: ${operation.cost_estimate.total_cost:.4f}"
        if operation.cost_estimate
        else "No cost estimate"
    )
    print()


def demo_error_tracking():
    """Demonstrate error tracking and categorization."""
    print("🚨 Demo 2: Error Tracking and Categorization")
    print("=" * 50)

    reset_metrics_collector()
    collector = get_metrics_collector()

    # Start operation
    operation_id = collector.start_operation("error_demo")

    # Simulate successful tool call
    collector.record_tool_call(
        operation_id=operation_id,
        tool_name="database_connection",
        status=ToolCallStatus.SUCCESS,
        duration=0.01,
    )

    # Simulate failed tool call
    collector.record_tool_call(
        operation_id=operation_id,
        tool_name="external_service",
        status=ToolCallStatus.FAILURE,
        duration=0.5,
        error_category=ErrorCategory.NETWORK,
        error_message="Connection timeout after 30 seconds",
    )

    # Simulate timeout
    collector.record_tool_call(
        operation_id=operation_id,
        tool_name="slow_api",
        status=ToolCallStatus.TIMEOUT,
        duration=5.0,
        error_category=ErrorCategory.TIMEOUT,
        error_message="Request timed out",
    )

    # End operation with errors
    operation = collector.end_operation(operation_id, success=False, error_count=2)

    print(f"Operation failed with {operation.error_count} errors")
    print(f"Tool calls: {len(operation.tool_calls)}")

    for tool_call in operation.tool_calls:
        status_emoji = "✅" if tool_call.status == ToolCallStatus.SUCCESS else "❌"
        print(f"  {status_emoji} {tool_call.tool_name}: {tool_call.status.value}")
        if tool_call.error_message:
            print(f"    Error: {tool_call.error_message}")

    print()


def demo_performance_trends():
    """Demonstrate performance trend analysis."""
    print("📈 Demo 3: Performance Trend Analysis")
    print("=" * 50)

    reset_metrics_collector()
    collector = get_metrics_collector()

    # Simulate multiple operations over time
    operation_types = ["qa_response", "vector_search", "llm_inference"]

    for i in range(10):
        for op_type in operation_types:
            operation_id = collector.start_operation(op_type)

            # Simulate varying performance (some improving, some declining)
            base_duration = 1.0 if op_type == "qa_response" else 0.5
            variation = (i % 3 - 1) * 0.1  # -0.1, 0, 0.1
            duration = max(0.1, base_duration + variation)

            time.sleep(duration * 0.1)  # Simulate work

            # Add some failures for trend analysis
            success = i < 8  # 80% success rate

            collector.end_operation(
                operation_id, success=success, error_count=0 if success else 1
            )

    # Analyze trends
    for op_type in operation_types:
        trend = collector.analyze_performance_trends(op_type, "1h")
        print(f"\n{op_type.upper()}:")
        print(f"  Data points: {trend.data_points}")
        print(f"  Avg duration: {trend.avg_duration:.3f}s")
        print(f"  Success rate: {trend.success_rate:.1f}%")
        print(
            f"  Trend: {trend.trend_direction} (strength: {trend.trend_strength:.2f})"
        )

        if trend.bottlenecks:
            print(f"  Bottlenecks: {', '.join(trend.bottlenecks)}")

        if trend.optimization_suggestions:
            print(f"  Suggestions: {', '.join(trend.optimization_suggestions)}")

    print()


def demo_context_manager():
    """Demonstrate the context manager for automatic metrics tracking."""
    print("🔄 Demo 4: Context Manager for Automatic Tracking")
    print("=" * 50)

    reset_metrics_collector()
    collector = get_metrics_collector()

    # Use context manager for automatic tracking
    with collector.operation_context(
        "context_demo", {"demo_type": "context_manager"}
    ) as operation_id:
        print(f"Inside context manager, operation ID: {operation_id}")

        # Simulate some work
        time.sleep(0.1)

        # Add some metrics
        collector.add_component_timing(operation_id, "setup", 0.02)
        collector.add_component_timing(operation_id, "processing", 0.05)
        collector.add_component_timing(operation_id, "cleanup", 0.03)

        print("Context manager will automatically end the operation when exiting")

    print("Context manager completed successfully")
    print()


def demo_decorator_usage():
    """Demonstrate the decorator for automatic metrics tracking."""
    print("🎯 Demo 5: Decorator for Automatic Tracking")
    print("=" * 50)

    reset_metrics_collector()

    # Define a function with metrics tracking
    @with_metrics_tracking("decorated_function")
    def process_data(data_size: int) -> str:
        """Simulate data processing with metrics tracking."""
        time.sleep(0.05)  # Simulate work

        if data_size > 1000:
            raise ValueError("Data size too large")

        return f"Processed {data_size} items"

    # Call the decorated function
    try:
        result = process_data(500)
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function failed: {e}")

    # Get metrics summary
    collector = get_metrics_collector()
    session_summary = collector.get_session_summary()

    print(f"Session operations: {session_summary.total_operations}")
    print(f"Successful operations: {session_summary.successful_operations}")
    print(f"Failed operations: {session_summary.failed_operations}")
    print()


def demo_persistence():
    """Demonstrate metrics persistence to disk."""
    print("💾 Demo 6: Metrics Persistence")
    print("=" * 50)

    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create some metrics
    with collector.operation_context("persistence_demo") as operation_id:
        time.sleep(0.1)
        collector.add_component_timing(operation_id, "data_load", 0.03)
        collector.add_component_timing(operation_id, "computation", 0.05)
        collector.add_component_timing(operation_id, "output", 0.02)

    # Save metrics to disk
    try:
        filepath = collector.save_metrics("demo_metrics.json")
        print(f"Metrics saved to: {filepath}")

        # Load metrics back
        loaded_metrics = collector.load_metrics(filepath)
        print(f"Loaded metrics session: {loaded_metrics.session_id}")
        print(f"Loaded operations: {len(loaded_metrics.operations)}")

    except Exception as e:
        print(f"Persistence demo failed: {e}")

    print()


def demo_system_metrics():
    """Demonstrate system metrics collection."""
    print("🖥️  Demo 7: System Metrics")
    print("=" * 50)

    collector = get_metrics_collector()
    system_metrics = collector.get_system_metrics()

    print("Current system metrics:")
    for key, value in system_metrics.items():
        if key == "memory_available":
            # Convert to MB for readability
            value_mb = value / (1024 * 1024)
            print(f"  {key}: {value_mb:.1f} MB")
        else:
            print(f"  {key}: {value}")

    print()


def demo_session_summary():
    """Demonstrate comprehensive session summary."""
    print("📊 Demo 8: Session Summary")
    print("=" * 50)

    collector = get_metrics_collector()
    session_summary = collector.get_session_summary()

    print("Session Summary:")
    print(f"  Session ID: {session_summary.session_id}")
    print(f"  Start time: {session_summary.start_time}")
    print(f"  Total operations: {session_summary.total_operations}")
    print(f"  Successful operations: {session_summary.successful_operations}")
    print(f"  Failed operations: {session_summary.failed_operations}")
    print(f"  Total duration: {session_summary.total_duration:.3f}s")
    print(f"  Avg operation duration: {session_summary.avg_operation_duration:.3f}s")
    print(f"  Total tokens: {session_summary.total_tokens}")
    print(f"  Total cost: ${session_summary.total_cost:.4f}")
    print(f"  Total tool calls: {session_summary.total_tool_calls}")
    print(f"  Successful tool calls: {session_summary.successful_tool_calls}")
    print(f"  Failed tool calls: {session_summary.failed_tool_calls}")

    if session_summary.trends:
        print(f"  Performance trends analyzed: {len(session_summary.trends)}")

    print()


def main():
    """Run all demos."""
    print("🚀 Performance Metrics Integration System Demo")
    print("=" * 60)
    print("This demo showcases Phase 5: Performance Metrics Integration")
    print("Features: Tool Call Metrics, Token Usage Tracking, Wall Time Measurement")
    print("=" * 60)
    print()

    try:
        demo_basic_metrics_collection()
        demo_error_tracking()
        demo_performance_trends()
        demo_context_manager()
        demo_decorator_usage()
        demo_persistence()
        demo_system_metrics()
        demo_session_summary()

        print("✅ All demos completed successfully!")
        print("\n🎉 Performance Metrics Integration System is ready for use!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

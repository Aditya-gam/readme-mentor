#!/usr/bin/env python3
"""
Demo script for Phase 5.2: Metrics Display Implementation.

This demo showcases the comprehensive metrics display functionality with:
- Normal Mode: Key metrics summary
- Verbose Mode: Detailed breakdown
- Debug Mode: Complete analysis with trends
- JSON Mode: Structured data for analysis

Usage:
    python demo_metrics_display.py
"""

import sys
import time
from pathlib import Path

from rich.console import Console

from app.logging.enums import OutputFormat, VerbosityLevel
from app.metrics import get_metrics_collector, reset_metrics_collector
from app.metrics.models import ErrorCategory, ToolCallStatus

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def demo_normal_mode():
    """Demo normal mode display (key metrics summary)."""
    print("🔧 Demo 1: Normal Mode Display")
    print("=" * 50)
    print("Shows key metrics summary with success rates, duration, and costs")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create some sample operations
    with collector.operation_context("demo_operation_1") as op_id:
        time.sleep(0.1)
        collector.add_component_timing(op_id, "data_processing", 0.05)
        collector.add_component_timing(op_id, "api_call", 0.03)
        collector.record_tool_call(
            op_id, "database_query", ToolCallStatus.SUCCESS, 0.02
        )
        collector.record_token_usage(op_id, 150, 75, "gpt-3.5-turbo")

    with collector.operation_context("demo_operation_2") as op_id:
        time.sleep(0.15)
        collector.add_component_timing(op_id, "validation", 0.02)
        collector.add_component_timing(op_id, "processing", 0.08)
        collector.record_tool_call(op_id, "external_api", ToolCallStatus.SUCCESS, 0.05)
        collector.record_token_usage(op_id, 200, 100, "gpt-3.5-turbo")

    # Display metrics in normal mode
    console = Console()
    collector.display_metrics(
        verbosity=VerbosityLevel.NORMAL,
        output_format=OutputFormat.RICH,
        console=console,
    )
    print()


def demo_verbose_mode():
    """Demo verbose mode display (detailed breakdown)."""
    print("📊 Demo 2: Verbose Mode Display")
    print("=" * 50)
    print("Shows detailed breakdown with component timing and tool calls")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create operations with more detailed metrics
    with collector.operation_context("qa_response") as op_id:
        time.sleep(0.2)
        collector.add_component_timing(op_id, "input_validation", 0.01)
        collector.add_component_timing(op_id, "vector_search", 0.05)
        collector.add_component_timing(op_id, "llm_inference", 0.12)
        collector.add_component_timing(op_id, "citation_processing", 0.02)
        collector.record_tool_call(
            op_id, "embedding_search", ToolCallStatus.SUCCESS, 0.05
        )
        collector.record_tool_call(op_id, "openai_api", ToolCallStatus.SUCCESS, 0.12)
        collector.record_token_usage(op_id, 500, 250, "gpt-3.5-turbo")

    with collector.operation_context("vector_search") as op_id:
        time.sleep(0.08)
        collector.add_component_timing(op_id, "query_processing", 0.02)
        collector.add_component_timing(op_id, "similarity_search", 0.04)
        collector.add_component_timing(op_id, "result_ranking", 0.02)
        collector.record_tool_call(op_id, "chroma_search", ToolCallStatus.SUCCESS, 0.04)
        collector.record_token_usage(op_id, 100, 0, "embedding-model")

    # Display metrics in verbose mode
    console = Console()
    collector.display_metrics(
        verbosity=VerbosityLevel.VERBOSE,
        output_format=OutputFormat.RICH,
        console=console,
    )
    print()


def demo_debug_mode():
    """Demo debug mode display (complete analysis with trends)."""
    print("🔍 Demo 3: Debug Mode Display")
    print("=" * 50)
    print("Shows complete analysis with trends and optimization suggestions")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create multiple operations to generate trends
    operation_types = ["qa_response", "vector_search", "llm_inference"]

    for i in range(5):
        for op_type in operation_types:
            with collector.operation_context(op_type) as op_id:
                # Simulate varying performance
                base_duration = 0.2 if op_type == "qa_response" else 0.1
                variation = (i % 3 - 1) * 0.05  # -0.05, 0, 0.05
                duration = max(0.05, base_duration + variation)

                time.sleep(duration)

                # Add component timing
                if op_type == "qa_response":
                    collector.add_component_timing(op_id, "validation", 0.01)
                    collector.add_component_timing(op_id, "search", duration * 0.3)
                    collector.add_component_timing(op_id, "inference", duration * 0.6)
                    collector.add_component_timing(op_id, "formatting", 0.02)
                elif op_type == "vector_search":
                    collector.add_component_timing(op_id, "query_processing", 0.02)
                    collector.add_component_timing(
                        op_id, "similarity_search", duration * 0.7
                    )
                    collector.add_component_timing(op_id, "ranking", 0.02)
                else:  # llm_inference
                    collector.add_component_timing(op_id, "tokenization", 0.01)
                    collector.add_component_timing(op_id, "generation", duration * 0.8)
                    collector.add_component_timing(op_id, "post_processing", 0.02)

                # Record tool calls
                if op_type == "qa_response":
                    collector.record_tool_call(
                        op_id, "vector_store", ToolCallStatus.SUCCESS, duration * 0.3
                    )
                    collector.record_tool_call(
                        op_id, "openai_api", ToolCallStatus.SUCCESS, duration * 0.6
                    )
                    collector.record_token_usage(
                        op_id, 300 + i * 50, 150 + i * 25, "gpt-3.5-turbo"
                    )
                elif op_type == "vector_search":
                    collector.record_tool_call(
                        op_id, "chroma_search", ToolCallStatus.SUCCESS, duration * 0.7
                    )
                else:
                    collector.record_tool_call(
                        op_id, "openai_api", ToolCallStatus.SUCCESS, duration * 0.8
                    )
                    collector.record_token_usage(
                        op_id, 200 + i * 30, 100 + i * 15, "gpt-3.5-turbo"
                    )

    # Display metrics in debug mode
    console = Console()
    collector.display_metrics(
        verbosity=VerbosityLevel.DEBUG,
        output_format=OutputFormat.RICH,
        console=console,
    )
    print()


def demo_json_mode():
    """Demo JSON mode display (structured data for analysis)."""
    print("📋 Demo 4: JSON Mode Display")
    print("=" * 50)
    print("Shows structured JSON data for programmatic analysis")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create sample operations
    with collector.operation_context("api_request") as op_id:
        time.sleep(0.1)
        collector.add_component_timing(op_id, "authentication", 0.02)
        collector.add_component_timing(op_id, "data_fetch", 0.06)
        collector.add_component_timing(op_id, "response_format", 0.02)
        collector.record_tool_call(op_id, "http_request", ToolCallStatus.SUCCESS, 0.06)
        collector.record_token_usage(op_id, 50, 25, "gpt-3.5-turbo")

    # Display metrics in JSON format with different verbosity levels
    console = Console()

    print("JSON Output (Normal Verbosity):")
    print("-" * 30)
    collector.display_metrics(
        verbosity=VerbosityLevel.NORMAL,
        output_format=OutputFormat.JSON,
        console=console,
    )
    print()

    print("JSON Output (Verbose):")
    print("-" * 30)
    collector.display_metrics(
        verbosity=VerbosityLevel.VERBOSE,
        output_format=OutputFormat.JSON,
        console=console,
    )
    print()


def demo_plain_mode():
    """Demo plain text mode display."""
    print("📝 Demo 5: Plain Text Mode Display")
    print("=" * 50)
    print("Shows metrics in plain text format")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create sample operations
    with collector.operation_context("file_processing") as op_id:
        time.sleep(0.12)
        collector.add_component_timing(op_id, "file_read", 0.03)
        collector.add_component_timing(op_id, "content_parse", 0.06)
        collector.add_component_timing(op_id, "metadata_extract", 0.03)
        collector.record_tool_call(op_id, "file_io", ToolCallStatus.SUCCESS, 0.03)
        collector.record_tool_call(op_id, "parser", ToolCallStatus.SUCCESS, 0.06)

    # Display metrics in plain text format
    console = Console()
    collector.display_metrics(
        verbosity=VerbosityLevel.VERBOSE,
        output_format=OutputFormat.PLAIN,
        console=console,
    )
    print()


def demo_error_tracking():
    """Demo error tracking in metrics display."""
    print("🚨 Demo 6: Error Tracking Display")
    print("=" * 50)
    print("Shows how errors are tracked and displayed in metrics")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create operations with some errors
    with collector.operation_context("successful_operation") as op_id:
        time.sleep(0.08)
        collector.add_component_timing(op_id, "processing", 0.06)
        collector.add_component_timing(op_id, "output", 0.02)
        collector.record_tool_call(op_id, "internal_api", ToolCallStatus.SUCCESS, 0.06)

    # Create failed operation manually
    failed_op_id = collector.start_operation("failed_operation")
    time.sleep(0.05)
    collector.add_component_timing(failed_op_id, "validation", 0.02)
    collector.add_component_timing(failed_op_id, "processing", 0.03)
    collector.record_tool_call(
        failed_op_id,
        "external_service",
        ToolCallStatus.FAILURE,
        0.03,
        error_category=ErrorCategory.NETWORK,
        error_message="Connection timeout",
    )
    # End with error
    collector.end_operation(failed_op_id, success=False, error_count=1)

    # Create timeout operation manually
    timeout_op_id = collector.start_operation("timeout_operation")
    time.sleep(0.1)
    collector.add_component_timing(timeout_op_id, "setup", 0.02)
    collector.add_component_timing(timeout_op_id, "long_running_task", 0.08)
    collector.record_tool_call(
        timeout_op_id,
        "slow_api",
        ToolCallStatus.TIMEOUT,
        0.08,
        error_category=ErrorCategory.TIMEOUT,
        error_message="Request timed out after 30 seconds",
    )
    # End with error
    collector.end_operation(timeout_op_id, success=False, error_count=1)

    # Display metrics showing error tracking
    console = Console()
    collector.display_metrics(
        verbosity=VerbosityLevel.VERBOSE,
        output_format=OutputFormat.RICH,
        console=console,
    )
    print()


def demo_file_loading():
    """Demo loading metrics from file."""
    print("💾 Demo 7: File Loading Display")
    print("=" * 50)
    print("Shows loading and displaying metrics from saved files")
    print()

    # Reset collector for clean demo
    reset_metrics_collector()
    collector = get_metrics_collector()

    # Create some operations and save them
    with collector.operation_context("saved_operation") as op_id:
        time.sleep(0.1)
        collector.add_component_timing(op_id, "data_load", 0.04)
        collector.add_component_timing(op_id, "computation", 0.04)
        collector.add_component_timing(op_id, "save", 0.02)
        collector.record_tool_call(op_id, "database", ToolCallStatus.SUCCESS, 0.04)
        collector.record_token_usage(op_id, 80, 40, "gpt-3.5-turbo")

    # Save metrics to file
    try:
        filepath = collector.save_metrics("demo_metrics.json")
        print(f"✅ Saved metrics to: {filepath}")

        # Load and display from file
        console = Console()
        loaded_metrics = collector.load_metrics(filepath)

        print("📊 Displaying loaded metrics:")
        print("-" * 30)

        # Use the display formatter directly
        from app.metrics.display import MetricsDisplayFormatter

        formatter = MetricsDisplayFormatter(console)
        formatter.display_metrics(
            loaded_metrics,
            verbosity=VerbosityLevel.NORMAL,
            output_format=OutputFormat.RICH,
        )

    except Exception as e:
        print(f"❌ Failed to save/load metrics: {e}")

    print()


def main():
    """Run all metrics display demos."""
    print("🚀 Phase 5.2: Metrics Display Implementation Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive metrics display functionality")
    print("Features: Normal, Verbose, Debug, and JSON display modes")
    print("=" * 60)
    print()

    try:
        demo_normal_mode()
        demo_verbose_mode()
        demo_debug_mode()
        demo_json_mode()
        demo_plain_mode()
        demo_error_tracking()
        demo_file_loading()

        print("✅ All metrics display demos completed successfully!")
        print("\n🎉 Metrics Display Implementation is ready for use!")
        print("\n📋 Available CLI commands:")
        print("  readme-mentor metrics                    # Display current session")
        print("  readme-mentor metrics --detailed         # Show detailed analysis")
        print("  readme-mentor metrics --verbosity 2      # Verbose output")
        print("  readme-mentor metrics --output-format json  # JSON output")
        print("  readme-mentor metrics --load-file <file> # Load from file")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo script showcasing enhanced output formats for readme-mentor.

This script demonstrates the three output formats (Rich, Plain, JSON) with
various operations including ingestion progress, Q&A sessions, performance
metrics, and error handling.
"""

import sys
import time
from pathlib import Path

from app.logging import setup_logging
from app.output import (
    ErrorFormatter,
    IngestionFormatter,
    PerformanceFormatter,
    QAFormatter,
)

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def demo_ingestion_progress(user_output):
    """Demo ingestion progress with different formats."""
    print("\n" + "=" * 60)
    print("DEMO: Ingestion Progress")
    print("=" * 60)

    # Create formatter
    ingestion_formatter = IngestionFormatter(user_output)

    # Start ingestion
    ingestion_formatter.start_ingestion(
        "https://github.com/octocat/Hello-World",
        {"chunk_size": 1024, "chunk_overlap": 128},
    )

    # Simulate file processing
    files = ["README.md", "docs/installation.md", "docs/api.md", "docs/examples.md"]

    for i, file in enumerate(files, 1):
        ingestion_formatter.file_processing_progress(i, len(files), file)
        time.sleep(0.5)  # Simulate processing delay

    # Simulate embedding progress
    for i in range(0, 50, 10):
        ingestion_formatter.embedding_progress(i, 50)
        time.sleep(0.2)

    # Complete ingestion
    ingestion_formatter.ingestion_complete(
        "octocat_Hello-World_collection", 45.2, len(files), 156
    )


def demo_qa_session(user_output):
    """Demo Q&A session with different formats."""
    print("\n" + "=" * 60)
    print("DEMO: Q&A Session")
    print("=" * 60)

    # Create formatter
    qa_formatter = QAFormatter(user_output)

    # Start session
    qa_formatter.session_start("octocat_Hello-World")

    # Receive question
    question = "How do I install readme-mentor?"
    qa_formatter.question_received(question)

    # Generate answer
    answer = """To install readme-mentor, you can use pip:

1. Install from PyPI:
   pip install readme-mentor

2. Or install from source:
   git clone https://github.com/user/readme-mentor
   cd readme-mentor
   pip install -e .

The tool requires Python 3.11+ and will automatically install all dependencies."""

    citations = [
        {
            "file": "README.md",
            "start_line": 15,
            "end_line": 25,
            "content": "Installation instructions for readme-mentor...",
        },
        {
            "file": "docs/installation.md",
            "start_line": 5,
            "end_line": 12,
            "content": "Detailed installation guide with requirements...",
        },
    ]

    metadata = {
        "latency_ms": 1250,
        "total_exchanges": 3,
        "model_used": "gpt-4",
        "confidence_score": 0.92,
    }

    qa_formatter.answer_generated(question, answer, citations, metadata)

    # Session summary
    qa_formatter.session_summary(120.5, 3, "octocat_Hello-World")


def demo_performance_summary(user_output):
    """Demo performance summary with different formats."""
    print("\n" + "=" * 60)
    print("DEMO: Performance Summary")
    print("=" * 60)

    # Create formatter
    perf_formatter = PerformanceFormatter(user_output)

    # Start operation
    perf_formatter.operation_start("ingestion")

    # Complete operation with metrics
    additional_metrics = {
        "total_files": 12,
        "total_chunks": 156,
        "embedding_model": "all-MiniLM-L6-v2",
        "memory_usage": 245.8,
        "cpu_usage": 15.3,
    }

    perf_formatter.operation_complete("ingestion", 45.2, additional_metrics)

    # Performance summary
    metrics = {
        "ingestion_duration": 45.2,
        "total_files": 12,
        "total_chunks": 156,
        "embedding_model": "all-MiniLM-L6-v2",
        "memory_usage": 245.8,
        "cpu_usage": 15.3,
        "qa_session": {
            "total_questions": 5,
            "avg_response_time": 1.2,
            "total_tokens": 1250,
        },
    }

    perf_formatter.performance_summary(metrics)


def demo_error_handling(user_output):
    """Demo error handling with different formats."""
    print("\n" + "=" * 60)
    print("DEMO: Error Handling")
    print("=" * 60)

    # Create formatter
    error_formatter = ErrorFormatter(user_output)

    # Simulate different types of errors
    errors = [
        (
            "repository_validation",
            ValueError("Invalid repository URL: https://invalid-url"),
        ),
        ("network_operation", ConnectionError("Failed to connect to GitHub API")),
        ("repository_access", PermissionError("Access denied to repository")),
        (
            "repository_lookup",
            FileNotFoundError("Repository not found: user/nonexistent-repo"),
        ),
    ]

    for operation, error in errors:
        error_formatter.operation_error(operation, error, f"Demo {operation}")
        print()  # Add spacing between errors


def demo_format_comparison():
    """Demo all three output formats side by side."""
    formats = [("Rich", "rich"), ("Plain", "plain"), ("JSON", "json")]

    for format_name, output_format in formats:
        print(f"\n{'=' * 80}")
        print(f"OUTPUT FORMAT: {format_name.upper()}")
        print(f"{'=' * 80}")

        # Set up logging for this format
        user_output, _ = setup_logging(output_format=output_format)

        # Run all demos
        demo_ingestion_progress(user_output)
        demo_qa_session(user_output)
        demo_performance_summary(user_output)
        demo_error_handling(user_output)


def main():
    """Main demo function."""
    print("🚀 README-Mentor Enhanced Output Formats Demo")
    print("=" * 80)
    print("This demo showcases the three output formats:")
    print("• Rich: Beautiful, interactive terminal output with colors and formatting")
    print("• Plain: Simple, clean text output suitable for scripts and automation")
    print("• JSON: Structured, machine-readable output for integration")
    print("=" * 80)

    if len(sys.argv) > 1:
        # Demo specific format
        format_arg = sys.argv[1].lower()
        if format_arg not in ["rich", "plain", "json"]:
            print(f"Unknown format: {format_arg}")
            print("Available formats: rich, plain, json")
            return 1

        print(f"\nDemoing {format_arg.upper()} format:")
        user_output, _ = setup_logging(output_format=format_arg)

        demo_ingestion_progress(user_output)
        demo_qa_session(user_output)
        demo_performance_summary(user_output)
        demo_error_handling(user_output)
    else:
        # Demo all formats
        demo_format_comparison()

    print("\n" + "=" * 80)
    print("✅ Demo completed!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

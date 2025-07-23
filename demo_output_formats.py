#!/usr/bin/env python3
"""Demo script for enhanced output formats - Phase 2 Implementation.

This script demonstrates all the enhanced output formatting features
implemented in Phase 2, including:
- Rich format with progress indicators, status messages, structured displays, and interactive elements
- Plain text format with simple output, structured layout, error formatting, and performance display
- JSON format with machine-readable data, metadata inclusion, error structure, and progress tracking
"""

import json
import sys
import time
from pathlib import Path

from app.logging.config import LoggingConfig
from app.logging.enums import ColorMode, OutputFormat, VerbosityLevel
from app.output import OutputManager

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def demo_rich_format():
    """Demonstrate Rich format features."""
    print("\n" + "=" * 80)
    print("🎨 RICH FORMAT DEMONSTRATION")
    print("=" * 80)

    config = LoggingConfig(
        output_format=OutputFormat.RICH,
        user_output_level=VerbosityLevel.VERBOSE,
        log_color=ColorMode.TRUE,
    )

    output = OutputManager(config)

    # Demo 1: Progress Indicators
    print("\n📊 Progress Indicators Demo:")
    output.start_operation("File Processing")

    with output.progress_bar(total=10, description="Processing files") as progress:
        for i in range(10):
            time.sleep(0.2)  # Simulate work
            progress.update(
                0, completed=i + 1, description=f"Processing file_{i + 1}.md"
            )

    output.end_operation(
        "File Processing", {"files_processed": 10, "processing_rate": 5.0}
    )

    # Demo 2: Status Messages
    print("\n💬 Status Messages Demo:")
    output.print_success("Repository ingested successfully!")
    output.print_warning("Some files were skipped due to size limits")
    output.print_info("Using enhanced embedding model")
    output.print_error("Network timeout occurred", Exception("Connection timeout"))

    # Demo 3: Structured Display
    print("\n📋 Structured Display Demo:")

    # Performance metrics table
    performance_data = [
        {"metric": "Files Processed", "value": "150", "unit": "files"},
        {"metric": "Chunks Created", "value": "1,250", "unit": "chunks"},
        {"metric": "Processing Time", "value": "45.2", "unit": "seconds"},
        {"metric": "Memory Usage", "value": "256", "unit": "MB"},
        {"metric": "CPU Usage", "value": "85", "unit": "%"},
    ]
    output.print_table(performance_data, "Performance Metrics")

    # Demo 4: Interactive Elements
    print("\n🔄 Interactive Elements Demo:")

    with output.status_spinner("Loading repository data...", "dots"):
        time.sleep(2)

    with output.live_spinner("Generating embeddings...", "line"):
        time.sleep(2)

    # Demo 5: Q&A Session
    print("\n🤖 Q&A Session Demo:")
    output.qa.session_start("demo-repo")

    question = "How do I implement authentication in this project?"
    answer = "You can implement authentication using JWT tokens. First, install the required dependencies..."

    citations = [
        {
            "file": "src/auth/jwt.py",
            "start_line": 15,
            "end_line": 25,
            "content": "def create_jwt_token(user_id: str) -> str:",
            "relevance": 0.95,
        },
        {
            "file": "docs/authentication.md",
            "start_line": 10,
            "end_line": 20,
            "content": "## JWT Authentication Setup",
            "relevance": 0.88,
        },
    ]

    metadata = {"response_time": 1.2, "token_count": 150, "confidence_score": 0.92}

    output.qa.question_received(question)
    output.qa.answer_generated(question, answer, citations, metadata)

    output.qa.session_summary(30.5, 5, "demo-repo")


def demo_plain_format():
    """Demonstrate Plain text format features."""
    print("\n" + "=" * 80)
    print("📝 PLAIN TEXT FORMAT DEMONSTRATION")
    print("=" * 80)

    config = LoggingConfig(
        output_format=OutputFormat.PLAIN,
        user_output_level=VerbosityLevel.VERBOSE,
        log_color=ColorMode.FALSE,
    )

    output = OutputManager(config)

    # Demo 1: Simple Output
    print("\n📄 Simple Output Demo:")
    output.print_info("Starting plain text demonstration")
    output.print_success("Operation completed successfully")
    output.print_warning("Some warnings occurred")
    output.print_error("An error was encountered")

    # Demo 2: Structured Layout
    print("\n📋 Structured Layout Demo:")
    output.print_separator()

    # Table with structured layout
    table_data = [
        {"operation": "File Processing", "status": "Completed", "duration": "2.5s"},
        {
            "operation": "Embedding Generation",
            "status": "Completed",
            "duration": "15.2s",
        },
        {"operation": "Index Creation", "status": "Completed", "duration": "3.1s"},
    ]
    output.print_table(table_data, "Operation Summary")

    # Demo 3: Error Formatting
    print("\n❌ Error Formatting Demo:")
    try:
        raise ValueError("Invalid configuration parameter")
    except Exception as e:
        output.error.operation_error("Configuration Validation", e, "Loading settings")

    # Demo 4: Performance Display
    print("\n📊 Performance Display Demo:")
    metrics = {
        "processing": {
            "files_processed": 150,
            "chunks_created": 1250,
            "processing_time": 45.2,
        },
        "memory": {"peak_usage": 256, "average_usage": 180},
        "performance": {"throughput": 3.3, "efficiency": 85.5},
    }
    output.performance.performance_summary(metrics)


def demo_json_format():
    """Demonstrate JSON format features."""
    print("\n" + "=" * 80)
    print("🔧 JSON FORMAT DEMONSTRATION")
    print("=" * 80)

    config = LoggingConfig(
        output_format=OutputFormat.JSON,
        user_output_level=VerbosityLevel.VERBOSE,
        log_color=ColorMode.FALSE,
    )

    output = OutputManager(config)

    # Demo 1: Machine-Readable Data
    print("\n🤖 Machine-Readable Data Demo:")
    output.start_operation("Data Processing")

    # Simulate progress updates
    for i in range(5):
        progress_data = {
            "current": i + 1,
            "total": 5,
            "percentage": ((i + 1) / 5) * 100,
            "status": "processing",
        }
        print(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "type": "progress_update",
                    "data": progress_data,
                }
            )
        )
        time.sleep(0.5)

    output.end_operation(
        "Data Processing", {"items_processed": 1000, "processing_rate": 2000}
    )

    # Demo 2: Metadata Inclusion
    print("\n📊 Metadata Inclusion Demo:")
    metadata = {
        "operation_type": "repository_ingestion",
        "repository_url": "https://github.com/example/repo",
        "branch": "main",
        "commit_hash": "abc123def456",
        "file_patterns": ["*.md", "docs/**/*"],
        "excluded_patterns": ["*.log", "node_modules/**"],
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }

    print(
        json.dumps(
            {
                "timestamp": time.time(),
                "type": "operation_metadata",
                "metadata": metadata,
            }
        )
    )

    # Demo 3: Error Structure
    print("\n❌ Error Structure Demo:")
    try:
        raise ConnectionError("Failed to connect to GitHub API")
    except Exception as e:
        output.error.network_error("https://api.github.com", e)

    # Demo 4: Progress Tracking
    print("\n📈 Progress Tracking Demo:")
    progress_events = [
        {
            "type": "operation_start",
            "operation": "file_scanning",
            "timestamp": time.time(),
        },
        {"type": "progress_update", "current": 10, "total": 50, "percentage": 20},
        {"type": "progress_update", "current": 25, "total": 50, "percentage": 50},
        {"type": "progress_update", "current": 50, "total": 50, "percentage": 100},
        {"type": "operation_complete", "operation": "file_scanning", "duration": 5.2},
    ]

    for event in progress_events:
        print(json.dumps(event))


def demo_integration():
    """Demonstrate integration of all formats."""
    print("\n" + "=" * 80)
    print("🔗 INTEGRATION DEMONSTRATION")
    print("=" * 80)

    # Test all three formats with the same operation
    formats = [
        (OutputFormat.RICH, "Rich"),
        (OutputFormat.PLAIN, "Plain"),
        (OutputFormat.JSON, "JSON"),
    ]

    for output_format, format_name in formats:
        print(f"\n--- {format_name} Format ---")

        config = LoggingConfig(
            output_format=output_format,
            user_output_level=VerbosityLevel.VERBOSE,
            log_color=ColorMode.TRUE
            if output_format == OutputFormat.RICH
            else ColorMode.FALSE,
        )

        output = OutputManager(config)

        # Simulate a complete ingestion workflow
        output.start_operation("Repository Ingestion")

        # File processing
        with output.progress_bar(total=3, description="Processing files") as progress:
            for i in range(3):
                time.sleep(0.3)
                progress.update(
                    0, completed=i + 1, description=f"Processing file_{i + 1}.md"
                )

        # Add performance metrics
        output.add_performance_metric("files_processed", 3)
        output.add_performance_metric("chunks_created", 25)
        output.add_performance_metric("processing_time", 1.5)

        # Complete operation
        output.end_operation(
            "Repository Ingestion",
            {
                "files_processed": 3,
                "chunks_created": 25,
                "processing_time": 1.5,
                "memory_usage": 45.2,
            },
        )

        # Show performance summary
        output.print_performance_summary()


def main():
    """Run all demonstrations."""
    print("🚀 README-Mentor Enhanced Output Formats Demo - Phase 2")
    print("This demo showcases the enhanced output formatting features")
    print("implemented in Phase 2 of the README-Mentor project.")

    try:
        # Run all demos
        demo_rich_format()
        demo_plain_format()
        demo_json_format()
        demo_integration()

        print("\n" + "=" * 80)
        print("✅ All demonstrations completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

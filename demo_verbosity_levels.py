#!/usr/bin/env python3
"""Comprehensive demo of all verbosity levels for Phase 3 implementation.

This script demonstrates the complete verbosity level system with:
- Level 0 (QUIET): Only critical errors and final results
- Level 1 (NORMAL): Default level with success/failure status and basic metrics
- Level 2 (VERBOSE): Detailed operation steps and extended metrics
- Level 3 (DEBUG): All available information including raw data and internal state
"""

import random
import time

from app.logging import setup_logging
from app.logging.enums import OutputFormat, VerbosityLevel


def simulate_operation(
    user_output, operation_name: str, steps: list, duration: float = 2.0
):
    """Simulate an operation with multiple steps and tracking.

    Args:
        user_output: UserOutput instance
        operation_name: Name of the operation
        steps: List of step descriptions
        duration: Total duration for the operation
    """
    user_output.start_operation_timer(operation_name)

    # Simulate operation steps
    for i, step in enumerate(steps, 1):
        user_output.step(f"Step {i}: {step}", operation=operation_name)
        time.sleep(duration / len(steps))

    # Add some metrics
    user_output.add_token_count(operation_name, random.randint(100, 500))
    user_output.add_tool_call("github_api")
    user_output.add_tool_call("embedding_model")

    # End operation and print summary
    user_output.end_operation_timer(operation_name)
    user_output.print_operation_summary(operation_name)


def simulate_progress_bar(user_output, total_items: int, description: str):
    """Simulate a progress bar operation.

    Args:
        user_output: UserOutput instance
        total_items: Total number of items to process
        description: Description of the operation
    """
    with user_output.progress_bar(total_items, description) as progress:
        task = progress.add_task(description, total=total_items)

        for i in range(total_items):
            time.sleep(0.1)  # Simulate work
            progress.update(task, advance=1)

            # Add some detailed progress in verbose/debug modes
            if user_output.config.is_verbose():
                user_output.detail(f"Processed item {i + 1}/{total_items}")


def simulate_detailed_progress(
    user_output, total_items: int, description: str, operation: str
):
    """Simulate detailed progress tracking.

    Args:
        user_output: UserOutput instance
        total_items: Total number of items to process
        description: Description of the operation
        operation: Operation name for tracking
    """
    with user_output.detailed_progress_bar(
        total_items, description, operation
    ) as progress:
        task = progress.add_task(description, total=total_items)

        for i in range(total_items):
            time.sleep(0.1)  # Simulate work
            progress.update(task, advance=1)

            # Add operation steps
            user_output.step(f"Processing item {i + 1}", operation=operation)


def simulate_qa_session(user_output):
    """Simulate a Q&A session with different verbosity levels.

    Args:
        user_output: UserOutput instance
    """
    question = "How do I implement authentication in this project?"
    answer = "You can implement authentication using JWT tokens and middleware. The project supports multiple authentication providers including GitHub OAuth and local authentication."

    citations = [
        {
            "file": "app/auth/jwt.py",
            "start_line": 15,
            "end_line": 25,
            "content": "JWT token validation and generation functions",
        },
        {
            "file": "app/auth/oauth.py",
            "start_line": 45,
            "end_line": 60,
            "content": "GitHub OAuth integration and callback handling",
        },
    ]

    metadata = {
        "response_time": 1.2,
        "token_count": 150,
        "model_used": "gpt-4",
        "confidence_score": 0.95,
    }

    user_output.print_qa_session(question, answer, citations, metadata)


def simulate_error_handling(user_output):
    """Simulate error handling with different verbosity levels.

    Args:
        user_output: UserOutput instance
    """
    try:
        # Simulate an error
        raise ConnectionError("Failed to connect to GitHub API: Rate limit exceeded")
    except Exception as e:
        user_output.error(
            "Failed to fetch repository data",
            error=e,
            context="Repository ingestion process",
        )


def simulate_performance_metrics(user_output):
    """Simulate performance metrics collection and display.

    Args:
        user_output: UserOutput instance
    """
    # Add various performance metrics
    user_output.add_performance_metric("ingestion_duration", 15.5)
    user_output.add_performance_metric("qa_session_duration", 2.3)
    user_output.add_performance_metric("total_files", 45)
    user_output.add_performance_metric("total_chunks", 120)
    user_output.add_performance_metric("memory_usage", 256.5)
    user_output.add_performance_metric("cpu_usage", 12.3)

    # Add detailed metrics for verbose/debug modes
    if user_output.config.show_detailed_metrics:
        user_output.add_performance_metric("embedding_generation_time", 8.2)
        user_output.add_performance_metric("vector_search_time", 0.5)
        user_output.add_performance_metric("llm_inference_time", 1.8)

    # Add raw metrics for debug mode
    if user_output.config.show_raw_metrics:
        user_output.add_performance_metric("raw_network_latency", 45.2)
        user_output.add_performance_metric("raw_disk_io_time", 12.8)
        user_output.add_performance_metric("internal_cache_hits", 89)

    # Display metrics
    user_output.print_performance_metrics()


def simulate_configuration_display(user_output):
    """Simulate configuration details display.

    Args:
        user_output: UserOutput instance
    """
    config_details = {
        "model_name": "gpt-4",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    for key, value in config_details.items():
        user_output.config_detail(key, value)


def simulate_internal_state(user_output):
    """Simulate internal state display.

    Args:
        user_output: UserOutput instance
    """
    internal_state = {
        "vector_store": {
            "collection_name": "readme_mentor_collection",
            "document_count": 120,
            "embedding_dimension": 384,
            "index_type": "hnsw",
        },
        "llm_provider": {
            "provider": "openai",
            "model": "gpt-4",
            "api_version": "2024-01-01",
            "rate_limit_remaining": 9500,
        },
        "cache": {
            "cache_hits": 89,
            "cache_misses": 12,
            "cache_size": "45.2 MB",
            "eviction_count": 3,
        },
    }

    user_output.internal_state(internal_state)


def simulate_raw_data_display(user_output):
    """Simulate raw data display.

    Args:
        user_output: UserOutput instance
    """
    raw_data = {
        "api_response": {
            "status_code": 200,
            "headers": {
                "content-type": "application/json",
                "x-ratelimit-remaining": "9500",
                "x-ratelimit-reset": "1640995200",
            },
            "body": {
                "repository": {
                    "id": 123456,
                    "name": "readme-mentor",
                    "full_name": "Aditya-gam/readme-mentor",
                    "description": "AI-powered README generation tool",
                }
            },
        },
        # Truncated for display
        "embedding_vector": [0.123, -0.456, 0.789, ...],
        "chunk_metadata": {
            "file_path": "app/main.py",
            "start_line": 10,
            "end_line": 25,
            "chunk_id": "chunk_001",
            "embedding_id": "emb_001",
        },
    }

    user_output.raw(raw_data, "API Response and Processing Data")


def demo_verbosity_level(level: int, output_format: OutputFormat = OutputFormat.RICH):
    """Demonstrate a specific verbosity level.

    Args:
        level: Verbosity level (0-3)
        output_format: Output format to use
    """
    print(f"\n{'=' * 80}")
    print(f"DEMONSTRATING VERBOSITY LEVEL {level}")
    print(f"Output Format: {output_format.value.upper()}")
    print(f"{'=' * 80}")

    # Set up logging for this level
    user_output, dev_logger = setup_logging(
        user_output_level=str(level), output_format=output_format.value
    )

    # Display level information
    level_name = VerbosityLevel(level).name
    level_desc = VerbosityLevel(level).get_description()
    user_output.info(f"Verbosity Level: {level_name} ({level})")
    user_output.info(f"Description: {level_desc}")

    # Simulate different operations based on verbosity level
    if level >= 1:  # NORMAL and above
        user_output.success("Starting repository ingestion process")

        # Basic progress bar
        simulate_progress_bar(user_output, 10, "Fetching repository files")

        # Basic operation
        simulate_operation(
            user_output,
            "file_processing",
            ["Reading files", "Parsing content", "Generating embeddings"],
            1.5,
        )

        # Basic Q&A session
        simulate_qa_session(user_output)

        # Basic performance metrics
        simulate_performance_metrics(user_output)

    if level >= 2:  # VERBOSE and above
        user_output.verbose("Enabling detailed operation tracking")

        # Configuration details
        simulate_configuration_display(user_output)

        # Detailed progress with operation tracking
        simulate_detailed_progress(
            user_output, 15, "Processing documentation files", "doc_processing"
        )

        # More detailed operation
        simulate_operation(
            user_output,
            "embedding_generation",
            [
                "Loading embedding model",
                "Tokenizing text chunks",
                "Generating embeddings",
                "Storing in vector database",
                "Creating index",
            ],
            2.5,
        )

    if level >= 3:  # DEBUG only
        user_output.debug("Enabling full debug information")

        # Internal state
        simulate_internal_state(user_output)

        # Raw data display
        simulate_raw_data_display(user_output)

        # Most detailed operation
        simulate_operation(
            user_output,
            "full_pipeline",
            [
                "Initializing components",
                "Loading configuration",
                "Connecting to external services",
                "Processing input data",
                "Generating embeddings",
                "Storing results",
                "Cleaning up resources",
            ],
            3.0,
        )

    # Error handling (always shown, but with different detail levels)
    if level >= 1:
        simulate_error_handling(user_output)

    # Final summary
    if level >= 1:
        user_output.success("Demo completed successfully")
        user_output.info(f"Verbosity level {level} demonstration finished")


def main():
    """Run the complete verbosity level demonstration."""
    print("README-Mentor Phase 3: Verbosity Level Implementation Demo")
    print("This demo showcases all 4 verbosity levels with their specific features")

    # Test all verbosity levels with Rich format
    for level in range(4):
        demo_verbosity_level(level, OutputFormat.RICH)
        time.sleep(1)  # Brief pause between demos

    # Test with different output formats for level 2 (VERBOSE)
    print(f"\n{'=' * 80}")
    print("TESTING DIFFERENT OUTPUT FORMATS WITH VERBOSITY LEVEL 2")
    print(f"{'=' * 80}")

    for output_format in [OutputFormat.PLAIN, OutputFormat.JSON]:
        demo_verbosity_level(2, output_format)
        time.sleep(1)

    print(f"\n{'=' * 80}")
    print("VERBOSITY LEVEL DEMONSTRATION COMPLETED")
    print("Key Features Demonstrated:")
    print("• Level 0 (QUIET): Minimal output, errors only")
    print("• Level 1 (NORMAL): Success/failure status, basic metrics, progress bars")
    print("• Level 2 (VERBOSE): Detailed steps, configuration, extended metrics")
    print("• Level 3 (DEBUG): All information, internal state, raw data")
    print("• Multiple output formats: Rich, Plain, JSON")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

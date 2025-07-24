#!/usr/bin/env python3
"""Demo script for the Developer Error System.

This script demonstrates the comprehensive developer error handling capabilities
including technical error details, stack trace information, context data,
and debugging information with structured logging and metadata.
"""

import sys
import time
import traceback

from app.errors import (
    CodeExecutionError,
    DependencyError,
    DeveloperErrorCategory,
    DeveloperErrorCode,
    DeveloperErrorSeverity,
    IntegrationError,
    MemoryManagementError,
    PerformanceError,
    create_developer_error,
    get_developer_error_manager,
    handle_developer_exception,
)
from app.logging.formatters import (
    DeveloperErrorFormatter,
    DeveloperErrorLogFormatter,
    StructuredDeveloperErrorFormatter,
)

# Add the app directory to the path
sys.path.insert(0, ".")


def demo_basic_developer_error():
    """Demonstrate basic developer error creation."""
    print("\n" + "=" * 60)
    print("DEMO: Basic Developer Error Creation")
    print("=" * 60)

    # Create a developer error manager
    dev_error_manager = get_developer_error_manager()
    dev_error_manager.start_session(
        session_id="demo_session_001",
        developer_id="demo_user",
        environment="development",
    )

    # Create a simple developer error
    error = create_developer_error(
        error_code=DeveloperErrorCode.FUNCTION_CALL_FAILED,
        title="Demo Function Call Error",
        message="This is a demonstration of a basic developer error",
        category=DeveloperErrorCategory.CODE_EXECUTION,
        severity=DeveloperErrorSeverity.ERROR,
        debug_info={
            "demo_mode": True,
            "test_case": "basic_error_creation",
            "timestamp": time.time(),
        },
    )

    print("Created Developer Error:")
    print(f"  Error ID: {error.error_id}")
    print(f"  Error Code: {error.error_code}")
    print(f"  Category: {error.category}")
    print(f"  Severity: {error.severity}")
    print(f"  Title: {error.title}")
    print(f"  Message: {error.message}")
    print(f"  Created At: {error.created_at}")

    return error


def demo_exception_handling():
    """Demonstrate handling of actual exceptions."""
    print("\n" + "=" * 60)
    print("DEMO: Exception Handling")
    print("=" * 60)

    # Simulate different types of exceptions
    exceptions_to_test = [
        ("TypeError", TypeError("Invalid type for operation")),
        ("ValueError", ValueError("Invalid value provided")),
        ("AttributeError", AttributeError("Object has no attribute 'nonexistent'")),
        ("KeyError", KeyError("Key 'missing_key' not found")),
        ("IndexError", IndexError("List index out of range")),
    ]

    for exception_name, exception in exceptions_to_test:
        print(f"\nHandling {exception_name}:")
        try:
            # This will raise the exception
            raise exception
        except Exception as e:
            # Handle the exception with developer error system
            developer_error = handle_developer_exception(
                e,
                operation="demo_exception_handling",
                component="demo_script",
                function_name="demo_exception_handling",
            )

            print(f"  Error ID: {developer_error.error_id}")
            print(f"  Error Code: {developer_error.error_code}")
            print(f"  Category: {developer_error.category}")
            print(f"  Severity: {developer_error.severity}")
            print(f"  Exception Type: {developer_error.exception_type}")
            print(f"  Stack Frames: {len(developer_error.stack_frames)}")

            # Show first stack frame
            if developer_error.stack_frames:
                first_frame = developer_error.stack_frames[0]
                print(
                    f"  First Frame: {first_frame.filename}:{first_frame.line_number}"
                )


def demo_specialized_exceptions():
    """Demonstrate specialized developer exceptions."""
    print("\n" + "=" * 60)
    print("DEMO: Specialized Developer Exceptions")
    print("=" * 60)

    # Code Execution Error
    print("\n1. Code Execution Error:")
    code_error = CodeExecutionError(
        error_code=DeveloperErrorCode.INVALID_ARGUMENT_TYPE,
        title="Invalid Argument Type",
        message="Function received unexpected argument type",
        function_name="process_data",
        arguments={"data": "string", "expected": "list"},
    )
    print(f"  Error: {code_error}")
    print(f"  Function: {code_error.function_name}")
    print(f"  Arguments: {code_error.arguments}")

    # Memory Management Error
    print("\n2. Memory Management Error:")
    memory_error = MemoryManagementError(
        error_code=DeveloperErrorCode.MEMORY_LEAK_DETECTED,
        title="Memory Leak Detected",
        message="Potential memory leak in data processing",
        memory_usage={
            "rss": 1024 * 1024 * 100,  # 100MB
            "vms": 1024 * 1024 * 200,  # 200MB
            "percent": 85.5,
        },
    )
    print(f"  Error: {memory_error}")
    print(f"  Memory Usage: {memory_error.memory_usage}")

    # Performance Error
    print("\n3. Performance Error:")
    performance_error = PerformanceError(
        error_code=DeveloperErrorCode.TIMEOUT_EXCEEDED,
        title="Operation Timeout",
        message="Database query exceeded timeout limit",
        execution_time=45.7,
        resource_usage={"cpu_percent": 95.2, "memory_percent": 87.3, "disk_io": "high"},
    )
    print(f"  Error: {performance_error}")
    print(f"  Execution Time: {performance_error.execution_time}s")
    print(f"  Resource Usage: {performance_error.resource_usage}")

    # Integration Error
    print("\n4. Integration Error:")
    integration_error = IntegrationError(
        error_code=DeveloperErrorCode.API_CALL_FAILED,
        title="API Call Failed",
        message="External API service unavailable",
        service_name="GitHub API",
        endpoint="/repos/owner/repo",
        response_data={
            "status_code": 503,
            "error": "Service Unavailable",
            "retry_after": 60,
        },
    )
    print(f"  Error: {integration_error}")
    print(f"  Service: {integration_error.service_name}")
    print(f"  Endpoint: {integration_error.endpoint}")
    print(f"  Response: {integration_error.response_data}")

    # Dependency Error
    print("\n5. Dependency Error:")
    dependency_error = DependencyError(
        error_code=DeveloperErrorCode.IMPORT_ERROR,
        title="Missing Dependency",
        message="Required package not found",
        dependency_name="requests",
        required_version="2.25.0",
        installed_version="2.24.0",
    )
    print(f"  Error: {dependency_error}")
    print(f"  Dependency: {dependency_error.dependency_name}")
    print(f"  Required: {dependency_error.required_version}")
    print(f"  Installed: {dependency_error.installed_version}")


def demo_error_formatting():
    """Demonstrate different error formatting options."""
    print("\n" + "=" * 60)
    print("DEMO: Error Formatting")
    print("=" * 60)

    # Create a sample error
    error = create_developer_error(
        error_code=DeveloperErrorCode.FUNCTION_CALL_FAILED,
        title="Formatting Demo Error",
        message="This error demonstrates different formatting options",
        category=DeveloperErrorCategory.CODE_EXECUTION,
        severity=DeveloperErrorSeverity.WARNING,
        debug_info={
            "formatting_demo": True,
            "test_scenarios": ["json", "text", "rich", "structured"],
        },
    )

    # JSON Formatting
    print("\n1. JSON Formatting:")
    json_formatter = DeveloperErrorFormatter(format_type="json", include_metadata=True)
    json_output = json_formatter.format_developer_error(error)
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

    # Text Formatting
    print("\n2. Text Formatting:")
    text_formatter = DeveloperErrorFormatter(format_type="text", include_metadata=True)
    text_output = text_formatter.format_developer_error(error)
    print(text_output)

    # Structured Formatting
    print("\n3. Structured Formatting:")
    structured_formatter = StructuredDeveloperErrorFormatter(
        include_location=True, include_timestamp=True
    )
    structured_output = structured_formatter.format_error(error)
    print("Structured Error Data:")
    for key, value in structured_output.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


def demo_context_management():
    """Demonstrate error context management."""
    print("\n" + "=" * 60)
    print("DEMO: Context Management")
    print("=" * 60)

    dev_error_manager = get_developer_error_manager()

    # Use context manager for error tracking
    with dev_error_manager.developer_error_context(
        operation="data_processing",
        component="demo_script",
        function_name="demo_context_management",
        user_input={"file_path": "/path/to/data.csv"},
        environment={"python_version": "3.11.0", "platform": "darwin"},
    ) as context:
        print("Context created:")
        print(f"  Operation: {context.operation}")
        print(f"  Component: {context.component}")
        print(f"  Function: {context.function_name}")
        print(f"  Timestamp: {context.timestamp}")
        print(f"  Session ID: {context.session_id}")

        # Simulate an error within the context
        try:
            # This will raise an exception
            raise ValueError("Simulated error within context")
        except Exception as e:
            # The context manager will automatically handle this
            print(f"\nError occurred within context: {e}")
            # The exception will be re-raised with developer error information


def demo_error_statistics():
    """Demonstrate error statistics and reporting."""
    print("\n" + "=" * 60)
    print("DEMO: Error Statistics and Reporting")
    print("=" * 60)

    dev_error_manager = get_developer_error_manager()

    # Create some sample errors
    for i in range(5):
        create_developer_error(
            error_code=DeveloperErrorCode.FUNCTION_CALL_FAILED,
            title=f"Sample Error {i + 1}",
            message=f"This is sample error number {i + 1}",
            category=DeveloperErrorCategory.CODE_EXECUTION,
            severity=DeveloperErrorSeverity.ERROR,
        )

    # Get error statistics
    stats = dev_error_manager.get_developer_error_statistics()
    print("Error Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export error report
    report = dev_error_manager.export_developer_error_report(format="text")
    print("\nError Report Preview:")
    print(report[:300] + "..." if len(report) > 300 else report)


def demo_logging_integration():
    """Demonstrate logging integration."""
    print("\n" + "=" * 60)
    print("DEMO: Logging Integration")
    print("=" * 60)

    import logging

    # Create a custom logger with developer error formatter
    logger = logging.getLogger("developer_errors")
    logger.setLevel(logging.DEBUG)

    # Create handler with developer error formatter
    handler = logging.StreamHandler()
    formatter = DeveloperErrorLogFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        include_metadata=True,
        format_type="structured",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Create and log a developer error
    error = create_developer_error(
        error_code=DeveloperErrorCode.API_CALL_FAILED,
        title="Logging Demo Error",
        message="This error demonstrates logging integration",
        category=DeveloperErrorCategory.INTEGRATION,
        severity=DeveloperErrorSeverity.ERROR,
    )

    # Log the error
    logger.error("Developer error occurred", extra={"developer_error": error})

    print("Error logged with structured formatter. Check the output above.")


def main():
    """Run all developer error system demos."""
    print("Developer Error System Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive developer error handling system")
    print(
        "with technical details, stack traces, context data, and debugging information."
    )

    try:
        # Run all demos
        demo_basic_developer_error()
        demo_exception_handling()
        demo_specialized_exceptions()
        demo_error_formatting()
        demo_context_management()
        demo_error_statistics()
        demo_logging_integration()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The Developer Error System provides:")
        print("• Technical error details with comprehensive debugging information")
        print("• Stack trace analysis and structured frame data")
        print("• Context and state data capture")
        print("• Structured logging with metadata")
        print("• Error codes and categories for classification")
        print("• Location and timestamp information")
        print("• Related log entries and error correlation")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

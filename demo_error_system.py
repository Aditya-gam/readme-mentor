#!/usr/bin/env python3
"""Demo script for the user-facing error system.

This script demonstrates the comprehensive error handling system with different
error categories, suggestions, and output formats.
"""

import os
import sys
from pathlib import Path

from app.errors import get_error_manager, handle_exception
from app.errors.exceptions import (
    ConfigurationError,
    NetworkError,
    PermissionError,
    ResourceError,
    SystemError,
    ValidationError,
)
from app.logging import UserOutput
from app.logging.enums import OutputFormat, VerbosityLevel
from app.models import (
    ErrorCode,
)

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def demo_configuration_errors():
    """Demonstrate configuration error handling."""
    print("\n" + "=" * 60)
    print("🔧 CONFIGURATION ERRORS DEMO")
    print("=" * 60)

    error_manager = get_error_manager()

    # Demo missing API key
    print("\n1. Missing OpenAI API Key:")
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError(
                error_code=ErrorCode.MISSING_API_KEY,
                title="Missing OpenAI API Key",
                message="OpenAI API key is required for AI-powered features.",
                setting_name="OPENAI_API_KEY",
            )
    except ConfigurationError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"Category: {user_error.category}")
        print(f"Severity: {user_error.severity}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")
            if suggestion.command:
                print(f"    Command: {suggestion.command}")

    # Demo invalid configuration
    print("\n2. Invalid Configuration File:")
    try:
        raise ConfigurationError(
            error_code=ErrorCode.INVALID_CONFIG_FILE,
            title="Invalid Configuration File",
            message="The configuration file format is invalid or corrupted.",
            setting_name="config.yaml",
        )
    except ConfigurationError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")


def demo_network_errors():
    """Demonstrate network error handling."""
    print("\n" + "=" * 60)
    print("🌐 NETWORK ERRORS DEMO")
    print("=" * 60)

    # Demo connection timeout
    print("\n1. Connection Timeout:")
    try:
        raise NetworkError(
            error_code=ErrorCode.CONNECTION_TIMEOUT,
            title="Connection Timeout",
            message="The connection to the server timed out.",
            url="https://api.openai.com/v1/chat/completions",
        )
    except NetworkError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"URL: {e.url}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")

    # Demo rate limit
    print("\n2. Rate Limit Exceeded:")
    try:
        raise NetworkError(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            title="Rate Limit Exceeded",
            message="Too many requests have been made. Please wait before trying again.",
            url="https://api.github.com/repos/user/repo",
            retry_after=60,
        )
    except NetworkError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"Retry After: {user_error.retry_after} seconds")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")


def demo_permission_errors():
    """Demonstrate permission error handling."""
    print("\n" + "=" * 60)
    print("🔐 PERMISSION ERRORS DEMO")
    print("=" * 60)

    # Demo access denied
    print("\n1. Access Denied:")
    try:
        raise PermissionError(
            error_code=ErrorCode.ACCESS_DENIED,
            title="Access Denied",
            message="You don't have permission to access this resource.",
            resource="/data/private_repo",
            required_permissions=["read", "write"],
        )
    except PermissionError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"Resource: {e.resource}")
        print(f"Required Permissions: {e.required_permissions}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")

    # Demo private repository
    print("\n2. Private Repository:")
    try:
        raise PermissionError(
            error_code=ErrorCode.REPOSITORY_PRIVATE,
            title="Private Repository",
            message="This repository is private and requires authentication.",
            resource="https://github.com/user/private-repo",
        )
    except PermissionError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")


def demo_validation_errors():
    """Demonstrate validation error handling."""
    print("\n" + "=" * 60)
    print("✅ VALIDATION ERRORS DEMO")
    print("=" * 60)

    # Demo invalid repository URL
    print("\n1. Invalid Repository URL:")
    try:
        raise ValidationError(
            error_code=ErrorCode.INVALID_REPO_URL,
            title="Invalid Repository URL",
            message="The repository URL format is invalid.",
            field_name="repo_url",
            invalid_value="not-a-url",
            expected_format="https://github.com/owner/repo",
        )
    except ValidationError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"Field: {e.field_name}")
        print(f"Invalid Value: {e.invalid_value}")
        print(f"Expected Format: {e.expected_format}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")

    # Demo missing required field
    print("\n2. Missing Required Field:")
    try:
        raise ValidationError(
            error_code=ErrorCode.MISSING_REQUIRED_FIELD,
            title="Missing Required Field",
            message="A required field is missing from your input.",
            field_name="query",
            expected_format="non-empty string",
        )
    except ValidationError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"Field: {e.field_name}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")


def demo_resource_errors():
    """Demonstrate resource error handling."""
    print("\n" + "=" * 60)
    print("📁 RESOURCE ERRORS DEMO")
    print("=" * 60)

    # Demo repository not found
    print("\n1. Repository Not Found:")
    try:
        raise ResourceError(
            error_code=ErrorCode.REPOSITORY_NOT_FOUND,
            title="Repository Not Found",
            message="The specified repository could not be found.",
            resource_path="https://github.com/nonexistent/repo",
            resource_type="repository",
        )
    except ResourceError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"Resource Path: {e.resource_path}")
        print(f"Resource Type: {e.resource_type}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")

    # Demo vector store not found
    print("\n2. Vector Store Not Found:")
    try:
        raise ResourceError(
            error_code=ErrorCode.VECTOR_STORE_NOT_FOUND,
            title="Vector Store Not Found",
            message="The vector store for this repository has not been created yet.",
            resource_path="data/user_repo/chroma",
            resource_type="vector_store",
        )
    except ResourceError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")


def demo_system_errors():
    """Demonstrate system error handling."""
    print("\n" + "=" * 60)
    print("💻 SYSTEM ERRORS DEMO")
    print("=" * 60)

    # Demo memory error
    print("\n1. Insufficient Memory:")
    try:
        raise SystemError(
            error_code=ErrorCode.MEMORY_ERROR,
            title="Insufficient Memory",
            message="There is not enough memory available to complete the operation.",
            system_component="embedding_model",
        )
    except SystemError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print(f"System Component: {e.system_component}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")

    # Demo disk space error
    print("\n2. Insufficient Disk Space:")
    try:
        raise SystemError(
            error_code=ErrorCode.DISK_SPACE_ERROR,
            title="Insufficient Disk Space",
            message="There is not enough disk space available.",
            system_component="storage",
        )
    except SystemError as e:
        user_error = e.to_user_facing_error()
        print(f"Error Code: {user_error.error_code}")
        print(f"Title: {user_error.title}")
        print(f"Message: {user_error.message}")
        print("Suggestions:")
        for suggestion in user_error.suggestions:
            print(f"  • {suggestion.title}: {suggestion.description}")


def demo_error_formatter():
    """Demonstrate error formatter with different output formats."""
    print("\n" + "=" * 60)
    print("🎨 ERROR FORMATTER DEMO")
    print("=" * 60)

    # Create a sample error
    error = ConfigurationError(
        error_code=ErrorCode.MISSING_API_KEY,
        title="Missing OpenAI API Key",
        message="OpenAI API key is required for AI-powered features.",
        setting_name="OPENAI_API_KEY",
    )

    # Demo different output formats
    formats = [OutputFormat.RICH, OutputFormat.PLAIN, OutputFormat.JSON]

    for output_format in formats:
        print(f"\n--- {output_format.value.upper()} FORMAT ---")

        # Create user output with specific format
        config = type(
            "Config",
            (),
            {
                "output_format": output_format,
                "verbosity_level": VerbosityLevel.NORMAL,
                "show_error_details": True,
                "is_quiet": lambda: False,
                "should_use_color": lambda: True,
                "show_stack_traces": True,
                "show_actionable_suggestions": True,
            },
        )()
        user_output = UserOutput(config)

        # Display the error
        user_output.formatter.operation_error("Configuration Check", error)


def demo_error_manager():
    """Demonstrate error manager functionality."""
    print("\n" + "=" * 60)
    print("📊 ERROR MANAGER DEMO")
    print("=" * 60)

    error_manager = get_error_manager()

    # Start a session
    error_manager.start_session("demo_session")

    # Simulate some errors
    errors = [
        ConfigurationError(
            error_code=ErrorCode.MISSING_API_KEY,
            title="Missing API Key",
            message="API key not found",
            setting_name="OPENAI_API_KEY",
        ),
        NetworkError(
            error_code=ErrorCode.CONNECTION_TIMEOUT,
            title="Connection Timeout",
            message="Network timeout occurred",
            url="https://api.openai.com",
        ),
        ValidationError(
            error_code=ErrorCode.INVALID_REPO_URL,
            title="Invalid URL",
            message="Invalid repository URL",
            field_name="repo_url",
        ),
    ]

    # Handle each error
    for i, error in enumerate(errors, 1):
        print(f"\nError {i}:")
        user_error = error_manager.handle_exception(
            error, context={"operation": f"demo_operation_{i}", "component": "demo"}
        )
        print(f"  Code: {user_error.error_code}")
        print(f"  Category: {user_error.category}")
        print(f"  Severity: {user_error.severity}")
        print(f"  Suggestions: {len(user_error.suggestions)}")

    # End session and get report
    report = error_manager.end_session()
    print("\nSession Report:")
    print(f"  Session ID: {report.session_id}")
    print(f"  Total Errors: {len(report.errors)}")

    # Get statistics
    stats = error_manager.get_error_statistics()
    print("\nError Statistics:")
    print(f"  Total Errors: {stats['total_errors']}")
    print(f"  Categories: {stats['category_distribution']}")
    print(f"  Severities: {stats['severity_distribution']}")


def demo_automatic_error_handling():
    """Demonstrate automatic error handling with different exception types."""
    print("\n" + "=" * 60)
    print("🤖 AUTOMATIC ERROR HANDLING DEMO")
    print("=" * 60)

    # Demo with built-in exceptions
    exceptions = [
        (ValueError("Invalid repository URL format"), "URL Validation"),
        (ConnectionError("Failed to connect to GitHub API"), "Network Request"),
        (PermissionError("Access denied to private repository"), "Repository Access"),
        (FileNotFoundError("Configuration file not found"), "File Access"),
        (MemoryError("Not enough memory for operation"), "System Resource"),
    ]

    for exception, operation in exceptions:
        print(f"\n{operation}:")
        try:
            raise exception
        except Exception as e:
            user_error = handle_exception(
                e,
                context={
                    "operation": operation.lower().replace(" ", "_"),
                    "component": "demo",
                },
                operation=operation,
            )
            print(f"  Original Exception: {type(e).__name__}")
            print(f"  Error Code: {user_error.error_code}")
            print(f"  Category: {user_error.category}")
            print(f"  Title: {user_error.title}")
            print(f"  Suggestions: {len(user_error.suggestions)}")


def main():
    """Run all error system demos."""
    print("🚀 README-MENTOR USER-FACING ERROR SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive error handling system")
    print("with different error categories, suggestions, and formatting options.")

    # Run all demos
    demo_configuration_errors()
    demo_network_errors()
    demo_permission_errors()
    demo_validation_errors()
    demo_resource_errors()
    demo_system_errors()
    demo_error_formatter()
    demo_error_manager()
    demo_automatic_error_handling()

    print("\n" + "=" * 60)
    print("✅ ERROR SYSTEM DEMO COMPLETED")
    print("=" * 60)
    print("The user-facing error system provides:")
    print("• Clear, non-technical error messages")
    print("• Actionable suggestions with commands")
    print("• Multiple output formats (Rich, Plain, JSON)")
    print("• Error categorization and severity levels")
    print("• Automatic error handling and logging")
    print("• Context-aware suggestions")
    print("• Retry guidance and next steps")


if __name__ == "__main__":
    main()

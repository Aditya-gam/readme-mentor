"""
Performance test configuration for readme-mentor.

This module provides configuration for performance testing across different
environments (CI, local development, production) and defines acceptable
performance thresholds for various operations.
"""

import os
from typing import Any, Dict


class PerformanceConfig:
    """
    Configuration class for performance testing thresholds and settings.

    This class provides environment-specific performance thresholds and
    configuration options for the readme-mentor QA system.
    """

    # Environment detection
    IS_CI = os.getenv("CI", "false").lower() == "true"
    IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

    # Performance thresholds (in milliseconds)
    # These are environment-specific and can be overridden
    THRESHOLDS = {
        "e2e_qa_response": {
            "ci": 5000,  # More lenient in CI due to resource constraints
            "local": 3000,  # Stricter for local development
            "production": 2000,  # Most strict for production
        },
        "vector_search": {
            "ci": 1000,
            "local": 500,
            "production": 300,
        },
        "llm_response": {
            "ci": 4000,
            "local": 2500,
            "production": 1500,
        },
    }

    @classmethod
    def get_environment(cls) -> str:
        """
        Determine the current environment for performance testing.

        Returns:
            str: Environment name ('ci', 'local', or 'production')
        """
        if cls.IS_CI or cls.IS_GITHUB_ACTIONS:
            return "ci"
        elif os.getenv("ENVIRONMENT") == "production":
            return "production"
        else:
            return "local"

    @classmethod
    def get_threshold(cls, operation: str) -> int:
        """
        Get the performance threshold for a specific operation.

        Args:
            operation: The operation name (e.g., 'e2e_qa_response')

        Returns:
            int: Threshold in milliseconds

        Raises:
            ValueError: If operation is not found in thresholds
        """
        env = cls.get_environment()

        if operation not in cls.THRESHOLDS:
            raise ValueError(f"Unknown performance operation: {operation}")

        if env not in cls.THRESHOLDS[operation]:
            # Fallback to local if environment not found
            env = "local"

        return cls.THRESHOLDS[operation][env]

    @classmethod
    def should_strict_enforce(cls) -> bool:
        """
        Determine if performance thresholds should be strictly enforced.

        In CI, we typically want to monitor but not fail tests due to
        performance variability. In local/production, we may want stricter
        enforcement.

        Returns:
            bool: True if thresholds should be strictly enforced
        """
        env = cls.get_environment()

        # Allow override via environment variable
        strict_env = os.getenv("STRICT_PERFORMANCE", "").lower()
        if strict_env == "true":
            return True
        elif strict_env == "false":
            return False

        # Default behavior: strict in local/production, lenient in CI
        return env in ["local", "production"]

    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of the current performance configuration.

        Returns:
            Dict containing configuration summary
        """
        env = cls.get_environment()

        return {
            "environment": env,
            "is_ci": cls.IS_CI,
            "is_github_actions": cls.IS_GITHUB_ACTIONS,
            "strict_enforcement": cls.should_strict_enforce(),
            "thresholds": {
                operation: cls.get_threshold(operation)
                for operation in cls.THRESHOLDS.keys()
            },
        }


def format_performance_log(operation: str, actual_latency_ms: float) -> str:
    """
    Format a standardized performance log message.

    Args:
        operation: The operation being measured
        actual_latency_ms: The actual latency in milliseconds

    Returns:
        str: Formatted log message
    """
    threshold = PerformanceConfig.get_threshold(operation)
    status = "PASS" if actual_latency_ms < threshold else "FAIL"

    return (
        f"Performance {operation}: {actual_latency_ms:.2f}ms "
        f"(threshold: {threshold}ms) - {status}"
    )


def log_performance_metrics(operation: str, actual_latency_ms: float) -> None:
    """
    Log performance metrics in a format suitable for CI monitoring.

    Args:
        operation: The operation being measured
        actual_latency_ms: The actual latency in milliseconds
    """
    import logging

    logger = logging.getLogger(__name__)

    # Standard log
    log_message = format_performance_log(operation, actual_latency_ms)
    logger.info(log_message)

    # CI-friendly print output (for GitHub Actions)
    print(f"PERFORMANCE_{operation.upper()}: {actual_latency_ms:.2f}ms")
    print(
        f"PERFORMANCE_{operation.upper()}_THRESHOLD: {PerformanceConfig.get_threshold(operation)}ms"
    )
    print(
        f"PERFORMANCE_{operation.upper()}_STATUS: {'PASS' if actual_latency_ms < PerformanceConfig.get_threshold(operation) else 'FAIL'}"
    )

    # Enhanced CI monitoring with environment context
    config = PerformanceConfig.get_config_summary()
    print(f"PERFORMANCE_{operation.upper()}_ENVIRONMENT: {config['environment']}")
    print(
        f"PERFORMANCE_{operation.upper()}_STRICT_ENFORCEMENT: {config['strict_enforcement']}"
    )

    # Performance trend indicators
    threshold = PerformanceConfig.get_threshold(operation)
    if actual_latency_ms < threshold * 0.5:
        print(f"PERFORMANCE_{operation.upper()}_TREND: EXCELLENT (<50% of threshold)")
    elif actual_latency_ms < threshold * 0.8:
        print(f"PERFORMANCE_{operation.upper()}_TREND: GOOD (<80% of threshold)")
    elif actual_latency_ms < threshold:
        print(f"PERFORMANCE_{operation.upper()}_TREND: ACCEPTABLE (<100% of threshold)")
    else:
        print(
            f"PERFORMANCE_{operation.upper()}_TREND: NEEDS_ATTENTION (>=100% of threshold)"
        )

    # Warning if threshold exceeded
    if actual_latency_ms >= PerformanceConfig.get_threshold(operation):
        logger.warning(
            f"Performance threshold exceeded for {operation}: "
            f"{actual_latency_ms:.2f}ms >= {PerformanceConfig.get_threshold(operation)}ms"
        )


def test_performance_config_basic():
    """
    Basic test for performance configuration functionality.
    """

    # Test environment detection
    env = PerformanceConfig.get_environment()
    assert env in ["ci", "local", "production"], f"Invalid environment: {env}"

    # Test threshold retrieval
    e2e_threshold = PerformanceConfig.get_threshold("e2e_qa_response")
    assert isinstance(e2e_threshold, int), "Threshold should be an integer"
    assert e2e_threshold > 0, "Threshold should be positive"

    # Test configuration summary
    config = PerformanceConfig.get_config_summary()
    assert "environment" in config, "Config should include environment"
    assert "thresholds" in config, "Config should include thresholds"

    # Test strict enforcement logic
    strict = PerformanceConfig.should_strict_enforce()
    assert isinstance(strict, bool), "Strict enforcement should be boolean"

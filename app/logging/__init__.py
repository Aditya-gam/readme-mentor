"""Logging module for README-Mentor.

This module provides structured logging capabilities with support for both
user-facing output and developer logging. It includes Rich-based user interfaces,
configurable verbosity levels, and performance metrics tracking.
"""

from typing import Optional

from .channels import DeveloperLogger, UserOutput
from .config import LoggingConfig, get_logging_config
from .enums import OutputFormat, VerbosityLevel
from .formatters import LogFormatter, OutputFormatter
from .handlers import CustomHandler

__all__ = [
    "LoggingConfig",
    "get_logging_config",
    "UserOutput",
    "DeveloperLogger",
    "OutputFormatter",
    "LogFormatter",
    "CustomHandler",
    "VerbosityLevel",
    "OutputFormat",
]


def setup_logging(
    log_level: Optional[str] = None,
    user_output_level: Optional[str] = None,
    output_format: Optional[str] = None,
    log_color: Optional[str] = None,
) -> tuple[UserOutput, DeveloperLogger]:
    """Initialize and configure logging system.

    Args:
        log_level: Developer log verbosity (DEBUG, INFO, WARNING, ERROR)
        user_output_level: User interface verbosity (QUIET, NORMAL, VERBOSE, DEBUG)
        output_format: Output format (RICH, PLAIN, JSON)
        log_color: Color output control (TRUE, FALSE, AUTO)

    Returns:
        Tuple of (UserOutput, DeveloperLogger) instances
    """
    config = get_logging_config()

    # Override config with provided parameters
    if log_level:
        config.log_level = log_level
    if user_output_level:
        config.user_output_level = user_output_level
    if output_format:
        config.output_format = output_format
    if log_color:
        config.log_color = log_color

    # Initialize output channels
    user_output = UserOutput(config)
    developer_logger = DeveloperLogger(config)

    return user_output, developer_logger


def get_user_output() -> UserOutput:
    """Get the global user output instance."""
    config = get_logging_config()
    return UserOutput(config)


def get_developer_logger() -> DeveloperLogger:
    """Get the global developer logger instance."""
    config = get_logging_config()
    return DeveloperLogger(config)

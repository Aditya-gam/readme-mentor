"""Enums for the logging system.

This module contains all enum definitions used by the logging system
to avoid circular import issues.
"""

from enum import Enum


class VerbosityLevel(Enum):
    """Verbosity levels for user output."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


class OutputFormat(Enum):
    """Output format options."""

    RICH = "rich"
    PLAIN = "plain"
    JSON = "json"


class LogLevel(Enum):
    """Standard logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColorMode(Enum):
    """Color output control modes."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    AUTO = "AUTO"

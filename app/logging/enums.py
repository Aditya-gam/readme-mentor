"""Enums for the logging system.

This module contains all enum definitions used by the logging system
to avoid circular import issues.
"""

from enum import Enum, IntEnum


class VerbosityLevel(IntEnum):
    """Verbosity levels for user output.

    Phase 3 implementation with 4 distinct levels:
    - QUIET (0): Only critical errors and final results
    - NORMAL (1): Default level with success/failure status and basic metrics
    - VERBOSE (2): Detailed operation steps and extended metrics
    - DEBUG (3): All available information including raw data and internal state
    """

    QUIET = 0  # Level 0: Only critical errors and final results
    NORMAL = 1  # Level 1: Default - success/failure status, basic metrics
    VERBOSE = 2  # Level 2: Detailed operation steps, extended metrics
    DEBUG = 3  # Level 3: All information, raw data, internal state

    @classmethod
    def from_string(cls, value: str) -> "VerbosityLevel":
        """Convert string to VerbosityLevel enum.

        Args:
            value: String representation of verbosity level

        Returns:
            VerbosityLevel enum value

        Raises:
            ValueError: If value is not a valid verbosity level
        """
        value_lower = value.lower()

        # Handle string mappings
        if value_lower in ["quiet", "0", "silent"]:
            return cls.QUIET
        elif value_lower in ["normal", "1", "default"]:
            return cls.NORMAL
        elif value_lower in ["verbose", "2", "detailed"]:
            return cls.VERBOSE
        elif value_lower in ["debug", "3", "all"]:
            return cls.DEBUG
        else:
            raise ValueError(f"Invalid verbosity level: {value}")

    def to_string(self) -> str:
        """Convert VerbosityLevel enum to string representation.

        Returns:
            String representation of the verbosity level
        """
        return self.name.lower()

    def get_description(self) -> str:
        """Get human-readable description of the verbosity level.

        Returns:
            Description of what this level includes
        """
        descriptions = {
            self.QUIET: "Only critical errors and final results",
            self.NORMAL: "Success/failure status, basic metrics, progress indicators",
            self.VERBOSE: "Detailed operation steps, extended metrics, configuration details",
            self.DEBUG: "All available information, raw data, internal state, full analysis",
        }
        return descriptions[self]

    def should_show_progress(self) -> bool:
        """Check if progress indicators should be shown at this level.

        Returns:
            True if progress should be shown
        """
        return self >= self.NORMAL

    def should_show_metrics(self) -> bool:
        """Check if performance metrics should be shown at this level.

        Returns:
            True if metrics should be shown
        """
        return self >= self.NORMAL

    def should_show_details(self) -> bool:
        """Check if detailed information should be shown at this level.

        Returns:
            True if details should be shown
        """
        return self >= self.VERBOSE

    def should_show_debug(self) -> bool:
        """Check if debug information should be shown at this level.

        Returns:
            True if debug info should be shown
        """
        return self >= self.DEBUG


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

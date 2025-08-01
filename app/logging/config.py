"""Centralized configuration management for logging system.

This module handles configuration loading from environment variables,
command-line arguments, and provides per-command configuration overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .enums import ColorMode, LogLevel, OutputFormat, VerbosityLevel


@dataclass
class LoggingConfig:
    """Configuration for the logging system.

    This class holds all configuration options for both user output
    and developer logging, with support for environment variable
    overrides and per-command customization.

    Phase 3 implementation supports 4 verbosity levels (0-3):
    - QUIET (0): Only critical errors and final results
    - NORMAL (1): Default level with success/failure status and basic metrics
    - VERBOSE (2): Detailed operation steps and extended metrics
    - DEBUG (3): All available information including raw data and internal state
    """

    # Core logging settings
    log_level: LogLevel = LogLevel.INFO
    user_output_level: VerbosityLevel = VerbosityLevel.NORMAL
    output_format: OutputFormat = OutputFormat.RICH
    log_color: ColorMode = ColorMode.AUTO

    # Performance metrics - controlled by verbosity level
    show_performance_metrics: bool = True
    track_tool_calls: bool = True
    track_token_counts: bool = True
    track_wall_time: bool = True
    show_detailed_metrics: bool = False  # For verbose/debug levels
    show_raw_metrics: bool = False  # For debug level only

    # Error handling
    show_stack_traces: bool = False
    show_actionable_suggestions: bool = True
    error_context_lines: int = 3
    show_error_details: bool = False  # For verbose/debug levels

    # Rich UI settings - controlled by verbosity level
    show_progress_bars: bool = True
    show_status_indicators: bool = True
    show_spinners: bool = True
    show_detailed_progress: bool = False  # For verbose/debug levels

    # Verbosity-specific settings
    show_operation_steps: bool = False  # For verbose/debug levels
    show_configuration_details: bool = False  # For verbose/debug levels
    show_internal_state: bool = False  # For debug level only
    show_raw_data: bool = False  # For debug level only

    # Per-command overrides
    command_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._apply_verbosity_settings()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.log_level, LogLevel):
            raise ValueError(f"Invalid log_level: {self.log_level}")

        if not isinstance(self.user_output_level, VerbosityLevel):
            raise ValueError(f"Invalid user_output_level: {self.user_output_level}")

        if not isinstance(self.output_format, OutputFormat):
            raise ValueError(f"Invalid output_format: {self.output_format}")

        if not isinstance(self.log_color, ColorMode):
            raise ValueError(f"Invalid log_color: {self.log_color}")

    def _apply_verbosity_settings(self) -> None:
        """Apply verbosity-specific settings based on user_output_level."""
        level = self.user_output_level

        # Level 0 (QUIET) - Minimal output
        if level == VerbosityLevel.QUIET:
            self.show_progress_bars = False
            self.show_status_indicators = False
            self.show_spinners = False
            self.show_performance_metrics = False
            self.show_actionable_suggestions = False
            self.show_operation_steps = False
            self.show_configuration_details = False
            self.show_internal_state = False
            self.show_raw_data = False
            self.show_detailed_metrics = False
            self.show_raw_metrics = False
            self.show_error_details = False
            self.show_detailed_progress = False

        # Level 1 (NORMAL) - Default behavior
        elif level == VerbosityLevel.NORMAL:
            self.show_progress_bars = True
            self.show_status_indicators = True
            self.show_spinners = True
            self.show_performance_metrics = True
            self.show_actionable_suggestions = True
            self.show_operation_steps = False
            self.show_configuration_details = False
            self.show_internal_state = False
            self.show_raw_data = False
            self.show_detailed_metrics = False
            self.show_raw_metrics = False
            self.show_error_details = False
            self.show_detailed_progress = False

        # Level 2 (VERBOSE) - Detailed information
        elif level == VerbosityLevel.VERBOSE:
            self.show_progress_bars = True
            self.show_status_indicators = True
            self.show_spinners = True
            self.show_performance_metrics = True
            self.show_actionable_suggestions = True
            self.show_operation_steps = True
            self.show_configuration_details = True
            self.show_internal_state = False
            self.show_raw_data = False
            self.show_detailed_metrics = True
            self.show_raw_metrics = False
            self.show_error_details = True
            self.show_detailed_progress = True

        # Level 3 (DEBUG) - All information
        elif level == VerbosityLevel.DEBUG:
            self.show_progress_bars = True
            self.show_status_indicators = True
            self.show_spinners = True
            self.show_performance_metrics = True
            self.show_actionable_suggestions = True
            self.show_operation_steps = True
            self.show_configuration_details = True
            self.show_internal_state = True
            self.show_raw_data = True
            self.show_detailed_metrics = True
            self.show_raw_metrics = True
            self.show_error_details = True
            self.show_detailed_progress = True
            self.show_stack_traces = True

    def get_command_config(self, command: str) -> "LoggingConfig":
        """Get configuration with command-specific overrides.

        Args:
            command: Command name (e.g., 'ingest', 'qa')

        Returns:
            New LoggingConfig instance with command overrides applied
        """
        if command not in self.command_overrides:
            return self

        # Create a copy with overrides
        config_dict = self.__dict__.copy()
        config_dict.update(self.command_overrides[command])

        # Remove command_overrides to avoid recursion
        config_dict.pop("command_overrides", None)

        return LoggingConfig(**config_dict)

    def set_command_override(self, command: str, **overrides: Any) -> None:
        """Set command-specific configuration overrides.

        Args:
            command: Command name
            **overrides: Configuration overrides
        """
        if command not in self.command_overrides:
            self.command_overrides[command] = {}

        self.command_overrides[command].update(overrides)

    def is_verbose(self) -> bool:
        """Check if verbose output is enabled."""
        return self.user_output_level >= VerbosityLevel.VERBOSE

    def is_debug(self) -> bool:
        """Check if debug output is enabled."""
        return self.user_output_level >= VerbosityLevel.DEBUG

    def is_quiet(self) -> bool:
        """Check if quiet output is enabled."""
        return self.user_output_level == VerbosityLevel.QUIET

    def should_use_color(self) -> bool:
        """Determine if color output should be used."""
        if self.log_color == ColorMode.TRUE:
            return True
        elif self.log_color == ColorMode.FALSE:
            return False
        else:  # AUTO
            # Check if terminal supports color
            return _is_color_supported()

    def should_show_rich_ui(self) -> bool:
        """Check if Rich UI elements should be shown."""
        return (
            self.output_format == OutputFormat.RICH
            and not self.is_quiet()
            and self.should_use_color()
        )

    def get_developer_log_level(self) -> LogLevel:
        """Get the appropriate developer log level based on verbosity.

        Returns:
            LogLevel appropriate for the current verbosity level
        """
        level_mapping = {
            VerbosityLevel.QUIET: LogLevel.ERROR,
            VerbosityLevel.NORMAL: LogLevel.INFO,
            VerbosityLevel.VERBOSE: LogLevel.DEBUG,
            VerbosityLevel.DEBUG: LogLevel.DEBUG,
        }
        return level_mapping.get(self.user_output_level, LogLevel.INFO)


def _is_color_supported() -> bool:
    """Check if the current terminal supports color output."""
    # Check for common environment variables
    if "NO_COLOR" in os.environ:
        return False

    if "FORCE_COLOR" in os.environ:
        return True

    # Check terminal type
    term = os.environ.get("TERM", "")
    if term in ["dumb", "unknown"]:
        return False

    # Check if we're in a CI environment
    if any(ci_var in os.environ for ci_var in ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]):
        return False

    return True


def _parse_enum_value(enum_class: type, value: str) -> Any:
    """Parse string value to enum, with fallback to default."""
    try:
        if enum_class == VerbosityLevel:
            return VerbosityLevel.from_string(value)
        else:
            return enum_class(value.upper())
    except (ValueError, KeyError):
        # Return first enum value as default
        return list(enum_class)[0]


def _parse_bool_env(key: str) -> Optional[bool]:
    """Parse boolean environment variable."""
    if key in os.environ:
        return os.environ[key].lower() == "true"
    return None


def _parse_int_env(key: str) -> Optional[int]:
    """Parse integer environment variable."""
    if key in os.environ:
        try:
            return int(os.environ[key])
        except ValueError:
            pass
    return None


def _load_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}

    # Core settings
    if "LOG_LEVEL" in os.environ:
        config["log_level"] = _parse_enum_value(LogLevel, os.environ["LOG_LEVEL"])

    if "USER_OUTPUT_LEVEL" in os.environ:
        config["user_output_level"] = _parse_enum_value(
            VerbosityLevel, os.environ["USER_OUTPUT_LEVEL"]
        )

    if "OUTPUT_FORMAT" in os.environ:
        config["output_format"] = _parse_enum_value(
            OutputFormat, os.environ["OUTPUT_FORMAT"]
        )

    if "LOG_COLOR" in os.environ:
        config["log_color"] = _parse_enum_value(ColorMode, os.environ["LOG_COLOR"])

    # Performance metrics
    for key in [
        "SHOW_PERFORMANCE_METRICS",
        "TRACK_TOOL_CALLS",
        "TRACK_TOKEN_COUNTS",
        "TRACK_WALL_TIME",
        "SHOW_DETAILED_METRICS",
        "SHOW_RAW_METRICS",
    ]:
        value = _parse_bool_env(key)
        if value is not None:
            config[key.lower()] = value

    # Error handling
    for key in [
        "SHOW_STACK_TRACES",
        "SHOW_ACTIONABLE_SUGGESTIONS",
        "SHOW_ERROR_DETAILS",
    ]:
        value = _parse_bool_env(key)
        if value is not None:
            config[key.lower()] = value

    error_context = _parse_int_env("ERROR_CONTEXT_LINES")
    if error_context is not None:
        config["error_context_lines"] = error_context

    # Rich UI settings
    for key in [
        "SHOW_PROGRESS_BARS",
        "SHOW_STATUS_INDICATORS",
        "SHOW_SPINNERS",
        "SHOW_DETAILED_PROGRESS",
    ]:
        value = _parse_bool_env(key)
        if value is not None:
            config[key.lower()] = value

    # Verbosity-specific settings
    for key in [
        "SHOW_OPERATION_STEPS",
        "SHOW_CONFIGURATION_DETAILS",
        "SHOW_INTERNAL_STATE",
        "SHOW_RAW_DATA",
    ]:
        value = _parse_bool_env(key)
        if value is not None:
            config[key.lower()] = value

    return config


# Global configuration instance
_config: Optional[LoggingConfig] = None


def get_logging_config() -> LoggingConfig:
    """Get the global logging configuration instance.

    Returns:
        LoggingConfig instance with environment variables loaded
    """
    global _config

    if _config is None:
        env_config = _load_from_env()
        _config = LoggingConfig(**env_config)

    return _config


def reset_logging_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


def update_logging_config(**overrides: Any) -> None:
    """Update the global configuration with new values.

    Args:
        **overrides: Configuration overrides to apply
    """
    global _config

    if _config is None:
        _config = get_logging_config()

    for key, value in overrides.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

    _config._validate_config()
    _config._apply_verbosity_settings()

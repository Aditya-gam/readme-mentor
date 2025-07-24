"""Configuration management for readme-mentor application."""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .errors import get_error_manager
from .errors.exceptions import ConfigurationError
from .models import ErrorCategory, ErrorCode, ErrorSeverity
from .version import get_version

# Load environment variables from .env file if it exists
# This will not raise exceptions if .env file doesn't exist
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        """Initialize settings from environment variables."""
        # Application settings
        self.APP_NAME: str = os.getenv("APP_NAME", "readme-mentor")
        self.APP_VERSION: str = get_version()
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        # Server settings
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self.RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"

        # Database settings
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL", "sqlite:///./data/readme_mentor.db"
        )

        # API Keys
        self.GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

        # External services
        self.CHROMA_DB_HOST: str = os.getenv("CHROMA_DB_HOST", "localhost")
        self.CHROMA_DB_PORT: int = int(os.getenv("CHROMA_DB_PORT", "8000"))

        # Security
        self.SECRET_KEY: str = os.getenv("SECRET_KEY", "your_secret_key_here")
        self.ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        )

        # Development settings
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    def validate(self) -> List[ConfigurationError]:
        """Validate required settings and return any configuration errors.

        Returns:
            List of configuration errors found
        """
        errors = []
        error_manager = get_error_manager()

        # Check for required API keys
        if not self.OPENAI_API_KEY:
            error = error_manager.create_user_facing_error(
                error_code=ErrorCode.MISSING_API_KEY,
                title="Missing OpenAI API Key",
                message="OpenAI API key is required for AI-powered features. Please set the OPENAI_API_KEY environment variable.",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.HIGH,
                operation="configuration_validation",
                component="config",
                setting_name="OPENAI_API_KEY",
            )
            errors.append(
                ConfigurationError(
                    error_code=ErrorCode.MISSING_API_KEY,
                    title="Missing OpenAI API Key",
                    message="OpenAI API key is required for AI-powered features.",
                    setting_name="OPENAI_API_KEY",
                )
            )

        # Check for valid secret key
        if not self.SECRET_KEY or self.SECRET_KEY == "your_secret_key_here":
            error = error_manager.create_user_facing_error(
                error_code=ErrorCode.INVALID_SETTING_VALUE,
                title="Invalid Secret Key",
                message="SECRET_KEY must be set to a secure value in production. Please set a strong secret key.",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.HIGH,
                operation="configuration_validation",
                component="config",
                setting_name="SECRET_KEY",
            )
            errors.append(
                ConfigurationError(
                    error_code=ErrorCode.INVALID_SETTING_VALUE,
                    title="Invalid Secret Key",
                    message="SECRET_KEY must be set to a secure value in production.",
                    setting_name="SECRET_KEY",
                )
            )

        # Check for valid port number
        if not (1 <= self.PORT <= 65535):
            error = error_manager.create_user_facing_error(
                error_code=ErrorCode.INVALID_SETTING_VALUE,
                title="Invalid Port Number",
                message=f"Port number must be between 1 and 65535. Current value: {self.PORT}",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.MEDIUM,
                operation="configuration_validation",
                component="config",
                setting_name="PORT",
            )
            errors.append(
                ConfigurationError(
                    error_code=ErrorCode.INVALID_SETTING_VALUE,
                    title="Invalid Port Number",
                    message=f"Port number must be between 1 and 65535. Current value: {self.PORT}",
                    setting_name="PORT",
                )
            )

        # Check for required directories
        required_dirs = ["data", "cache", "logs"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    error = error_manager.create_user_facing_error(
                        error_code=ErrorCode.ACCESS_DENIED,
                        title="Permission Denied",
                        message=f"Cannot create required directory: {dir_path}. Please check permissions.",
                        category=ErrorCategory.PERMISSION,
                        severity=ErrorSeverity.HIGH,
                        operation="configuration_validation",
                        component="filesystem",
                        setting_name=dir_name,
                    )
                    errors.append(
                        ConfigurationError(
                            error_code=ErrorCode.ACCESS_DENIED,
                            title="Permission Denied",
                            message=f"Cannot create required directory: {dir_path}",
                            setting_name=dir_name,
                        )
                    )

        return errors

    def validate_for_operation(self, operation: str) -> List[ConfigurationError]:
        """Validate settings for a specific operation.

        Args:
            operation: Operation to validate for (e.g., 'ingest', 'qa', 'api')

        Returns:
            List of configuration errors found
        """
        errors = []

        if operation in ["ingest", "qa", "api"]:
            # These operations require OpenAI API key
            if not self.OPENAI_API_KEY:
                errors.append(
                    ConfigurationError(
                        error_code=ErrorCode.MISSING_API_KEY,
                        title="Missing OpenAI API Key",
                        message="OpenAI API key is required for AI-powered operations.",
                        setting_name="OPENAI_API_KEY",
                    )
                )

        if operation in ["ingest"]:
            # Ingestion might benefit from GitHub token for private repos
            if not self.GITHUB_TOKEN:
                # This is a warning, not an error
                pass  # Could add a warning system here

        return errors


# Global settings instance
def get_settings() -> Settings:
    return Settings()


def validate_configuration() -> List[ConfigurationError]:
    """Validate the current configuration and return any errors.

    Returns:
        List of configuration errors found
    """
    settings = get_settings()
    return settings.validate()


def validate_configuration_for_operation(operation: str) -> List[ConfigurationError]:
    """Validate configuration for a specific operation.

    Args:
        operation: Operation to validate for

    Returns:
        List of configuration errors found
    """
    settings = get_settings()
    return settings.validate_for_operation(operation)

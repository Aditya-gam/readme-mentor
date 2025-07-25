"""Configuration management for readme-mentor application."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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

    def validate(self) -> None:
        """Validate required settings."""
        if not self.SECRET_KEY or self.SECRET_KEY == "your_secret_key_here":
            raise ValueError("SECRET_KEY must be set in environment variables")


# Global settings instance
def get_settings() -> Settings:
    return Settings()

"""Configuration management for readme-mentor application."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# This will not raise exceptions if .env file doesn't exist
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings:
    """Application settings loaded from environment variables."""

    # Application settings
    APP_NAME: str = os.getenv("APP_NAME", "readme-mentor")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.0.5")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"

    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/readme_mentor.db")

    # API Keys
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # External services
    CHROMA_DB_HOST: str = os.getenv("CHROMA_DB_HOST", "localhost")
    CHROMA_DB_PORT: int = int(os.getenv("CHROMA_DB_PORT", "8000"))

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_secret_key_here")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # Development settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    @classmethod
    def validate(cls) -> None:
        """Validate required settings."""
        if not cls.SECRET_KEY or cls.SECRET_KEY == "your_secret_key_here":
            raise ValueError("SECRET_KEY must be set in environment variables")


# Global settings instance
settings = Settings()

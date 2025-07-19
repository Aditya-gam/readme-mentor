"""
Readme Mentor - AI-powered README generation and mentoring tool.

This module initializes the application and loads environment variables.
"""

import logging
from typing import Optional

from .version import __version__, get_version

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.warning("python-dotenv not available, using system environment variables")
except Exception as e:
    logging.warning(
        f"Failed to load .env file: {e}. Using system environment variables"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Version information

__author__ = "Aditya Gambhir"
__email__ = "67105262+Aditya-gam@users.noreply.github.com"


def get_environment() -> Optional[str]:
    """Get the current environment setting."""
    import os

    return os.getenv("ENVIRONMENT", "development")

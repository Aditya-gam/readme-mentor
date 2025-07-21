import logging
import os

import requests
from langchain_core.language_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chat_model() -> BaseChatModel:
    """
    Dynamically selects and returns a chat model based on availability.

    Checks for a local Ollama model first, then falls back to OpenAI.
    Raises a RuntimeError if no backend is available.
    """
    # Check for Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if any(model["name"] == "llama3:8b" for model in models):
                logger.info("Using local Llama 3 (8B) via Ollama")
                return ChatOllama(model="llama3:8b", streaming=True)
    except requests.ConnectionError:
        logger.info("Ollama not running, checking for OpenAI API key.")

    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Using OpenAI GPT-3.5-Turbo via API")
        return ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=True)

    raise RuntimeError(
        "No LLM backend available. Please run Ollama or set OPENAI_API_KEY."
    )

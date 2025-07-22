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

    Priority order:
    1. OpenAI API (if OPENAI_API_KEY is set)
    2. Local Ollama with Llama 3.1 (if Ollama is running)
    3. Raises RuntimeError if no backend is available

    Returns:
        BaseChatModel: Configured chat model instance

    Raises:
        RuntimeError: If no LLM backend is available
    """
    # Check for OpenAI API key first
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if (
        openai_api_key
        and openai_api_key.strip()
        and not openai_api_key.startswith("your_")
    ):
        logger.info("Using OpenAI GPT-3.5-Turbo via API")
        try:
            return ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=True)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI model: {e}")
            logger.info("Falling back to Ollama...")
    else:
        logger.info("No valid OpenAI API key found, checking for Ollama...")

    # Check for Ollama
    try:
        logger.info("Checking Ollama availability...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(
                f"Available Ollama models: {[model['name'] for model in models]}"
            )

            # Try to find a suitable Llama model
            suitable_models = [
                "llama3:8b",
                "llama3.1:latest",
                "llama3.1:8b",
                "llama3:latest",
                "llama2:latest",
                "llama2:7b",
            ]

            for model_name in suitable_models:
                if any(model["name"] == model_name for model in models):
                    logger.info(f"Using local {model_name} via Ollama")
                    return ChatOllama(model=model_name, streaming=True)

            # If no specific model found, try the first available model
            if models:
                first_model = models[0]["name"]
                logger.info(
                    f"No specific Llama model found, using first available: {first_model}"
                )
                return ChatOllama(model=first_model, streaming=True)

        else:
            logger.warning(f"Ollama API returned status code: {response.status_code}")

    except requests.ConnectionError:
        logger.info("Ollama not running or not accessible")
    except requests.Timeout:
        logger.warning("Ollama connection timed out")
    except Exception as e:
        logger.warning(f"Error checking Ollama: {e}")

    # If we get here, no backend is available
    raise RuntimeError(
        "No LLM backend available. Please either:\n"
        "1. Set a valid OPENAI_API_KEY environment variable, or\n"
        "2. Start Ollama and ensure a model is available (e.g., 'ollama pull llama3.1:latest')"
    )

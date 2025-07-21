import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from app.llm.provider import get_chat_model


@patch("requests.get")
def test_get_chat_model_ollama_success(mock_get):
    """Tests that ChatOllama is returned when the llama3:8b model is available."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama3:8b"}]}
    mock_get.return_value = mock_response

    model = get_chat_model()
    assert isinstance(model, ChatOllama)
    assert model.model == "llama3:8b"


@patch("requests.get")
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
def test_get_chat_model_openai_fallback(mock_get):
    """Tests that ChatOpenAI is returned when Ollama is unavailable but an API key is set."""
    mock_get.side_effect = requests.ConnectionError("Ollama not running")

    model = get_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-3.5-turbo-0125"


@patch("requests.get")
@patch.dict(os.environ, {}, clear=True)
def test_get_chat_model_no_backend(mock_get):
    """Tests that a RuntimeError is raised when no backend is available."""
    mock_get.side_effect = requests.ConnectionError("Ollama not running")

    with pytest.raises(RuntimeError, match="No LLM backend available"):
        get_chat_model()

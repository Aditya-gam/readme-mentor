import pytest
from langchain.prompts import PromptTemplate

from app.prompts import PROMPTS_DIR, get_prompts, load_system_prompt, load_user_prompt

# Define the expected content for the system prompt
EXPECTED_SYSTEM_PROMPT_CONTENT = "You are an assistant for a GitHub repository Q&A. Provide brief, factual answers with sources cited inline using the format <doc_0>, <doc_1>, etc. where the number corresponds to the source document provided in the context. Do not deviate from the repository content. Always include at least one citation when providing information from the source documents."


@pytest.fixture
def setup_prompts_dir(tmp_path):
    """
    Sets up a temporary prompts directory for testing.
    """
    original_prompts_dir = PROMPTS_DIR
    # Temporarily change PROMPTS_DIR to the tmp_path for testing
    # This requires modifying the module's PROMPTS_DIR directly, which is generally
    # not recommended but necessary for testing module-level path configurations.
    # A more robust solution might involve dependency injection for PROMPTS_DIR.
    # For this specific case, given the simplicity, direct modification with cleanup is acceptable.

    # Create temporary system.txt and user.txt
    (tmp_path / "system.txt").write_text(EXPECTED_SYSTEM_PROMPT_CONTENT)
    (tmp_path / "user.txt").write_text("")

    # Patch the PROMPTS_DIR in the app.prompts module
    import app.prompts

    app.prompts.PROMPTS_DIR = tmp_path
    yield
    # Restore original PROMPTS_DIR after tests
    app.prompts.PROMPTS_DIR = original_prompts_dir


def test_load_system_prompt(setup_prompts_dir):
    """
    Tests that load_system_prompt correctly loads the system prompt.
    """
    system_prompt = load_system_prompt()
    assert isinstance(system_prompt, PromptTemplate)
    assert system_prompt.template == EXPECTED_SYSTEM_PROMPT_CONTENT


def test_load_user_prompt(setup_prompts_dir):
    """
    Tests that load_user_prompt correctly loads the user prompt (empty).
    """
    user_prompt = load_user_prompt()
    assert isinstance(user_prompt, PromptTemplate)
    assert user_prompt.template == ""


def test_get_prompts(setup_prompts_dir):
    """
    Tests that get_prompts returns a dictionary with both prompts.
    """
    prompts = get_prompts()
    assert "system_prompt" in prompts
    assert "user_prompt" in prompts
    assert isinstance(prompts["system_prompt"], PromptTemplate)
    assert isinstance(prompts["user_prompt"], PromptTemplate)
    assert prompts["system_prompt"].template == EXPECTED_SYSTEM_PROMPT_CONTENT
    assert prompts["user_prompt"].template == ""

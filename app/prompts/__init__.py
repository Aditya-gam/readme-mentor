"""
Prompt loading utility for the application.
"""

from pathlib import Path

from langchain.prompts import PromptTemplate

PROMPTS_DIR = Path(__file__).parent


def load_system_prompt() -> PromptTemplate:
    """
    Loads the system prompt from system.txt.
    """
    system_prompt_path = PROMPTS_DIR / "system.txt"
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt_content = f.read().strip()
    return PromptTemplate.from_template(system_prompt_content)


def load_user_prompt() -> PromptTemplate:
    """
    Loads the user prompt from user.txt.
    """
    user_prompt_path = PROMPTS_DIR / "user.txt"
    with open(user_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_content = f.read().strip()
    return PromptTemplate.from_template(user_prompt_content)


def get_prompts():
    """
    Returns a dictionary containing the loaded system and user prompts.
    """
    return {
        "system_prompt": load_system_prompt(),
        "user_prompt": load_user_prompt(),
    }

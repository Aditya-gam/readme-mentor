from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator


class QuestionPayload(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query or question.")
    repo_id: str = Field(
        ...,
        min_length=1,
        description="The ID of the repository (e.g., 'owner_repo') to query.",
    )
    history: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="A list of prior QA pairs (human_message, ai_message).",
    )

    @field_validator("query", "repo_id")
    @classmethod
    def not_empty_or_whitespace(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Field cannot be empty or contain only whitespace.")
        return v

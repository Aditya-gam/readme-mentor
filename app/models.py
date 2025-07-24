from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class ErrorSeverity(str, Enum):
    """Error severity levels for user-facing error system."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for user-facing error system."""

    CONFIGURATION = "configuration"
    NETWORK = "network"
    PERMISSION = "permission"
    VALIDATION = "validation"
    RESOURCE = "resource"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    """Standardized error codes for user-facing error system."""

    # Configuration errors
    MISSING_API_KEY = "CONFIG_001"
    INVALID_CONFIG_FILE = "CONFIG_002"
    MISSING_REQUIRED_ENV = "CONFIG_003"
    INVALID_SETTING_VALUE = "CONFIG_004"

    # Network errors
    CONNECTION_TIMEOUT = "NETWORK_001"
    RATE_LIMIT_EXCEEDED = "NETWORK_002"
    DNS_RESOLUTION_FAILED = "NETWORK_003"
    SSL_CERTIFICATE_ERROR = "NETWORK_004"
    PROXY_ERROR = "NETWORK_005"

    # Permission errors
    ACCESS_DENIED = "PERM_001"
    INSUFFICIENT_PERMISSIONS = "PERM_002"
    TOKEN_EXPIRED = "PERM_003"
    TOKEN_INVALID = "PERM_004"
    REPOSITORY_PRIVATE = "PERM_005"

    # Validation errors
    INVALID_REPO_URL = "VALID_001"
    INVALID_FILE_PATTERN = "VALID_002"
    INVALID_INPUT_FORMAT = "VALID_003"
    MISSING_REQUIRED_FIELD = "VALID_004"
    FIELD_TOO_LONG = "VALID_005"

    # Resource errors
    REPOSITORY_NOT_FOUND = "RESOURCE_001"
    FILE_NOT_FOUND = "RESOURCE_002"
    VECTOR_STORE_NOT_FOUND = "RESOURCE_003"
    MODEL_NOT_AVAILABLE = "RESOURCE_004"

    # System errors
    MEMORY_ERROR = "SYSTEM_001"
    DISK_SPACE_ERROR = "SYSTEM_002"
    PROCESS_TIMEOUT = "SYSTEM_003"
    UNEXPECTED_ERROR = "SYSTEM_004"


class ErrorSuggestion(BaseModel):
    """Model for error resolution suggestions."""

    title: str = Field(..., description="Short title for the suggestion")
    description: str = Field(..., description="Detailed description of the suggestion")
    command: Optional[str] = Field(None, description="Command to run to fix the issue")
    url: Optional[str] = Field(None, description="URL for additional help")


class ErrorContext(BaseModel):
    """Model for error context information."""

    operation: str = Field(..., description="Operation that failed")
    component: str = Field(..., description="Component where error occurred")
    timestamp: str = Field(..., description="ISO timestamp of error")
    user_input: Optional[Dict[str, Any]] = Field(
        None, description="User input that caused error"
    )
    environment: Optional[Dict[str, str]] = Field(
        None, description="Environment information"
    )


class UserFacingError(BaseModel):
    """Comprehensive user-facing error model."""

    error_code: ErrorCode = Field(..., description="Standardized error code")
    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    title: str = Field(..., description="Human-readable error title")
    message: str = Field(..., description="Clear, non-technical error message")
    technical_details: Optional[str] = Field(
        None, description="Technical details for debugging"
    )
    suggestions: List[ErrorSuggestion] = Field(
        default_factory=list, description="Actionable suggestions"
    )
    context: ErrorContext = Field(..., description="Error context information")
    next_steps: List[str] = Field(
        default_factory=list, description="Next steps guidance"
    )
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")

    class Config:
        use_enum_values = True


class ErrorReport(BaseModel):
    """Model for error reporting and aggregation."""

    errors: List[UserFacingError] = Field(
        default_factory=list, description="List of errors"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Error summary statistics"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

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


# Developer Error System Models


class DeveloperErrorSeverity(str, Enum):
    """Developer error severity levels for technical debugging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DeveloperErrorCategory(str, Enum):
    """Developer error categories for technical debugging."""

    CODE_EXECUTION = "code_execution"
    MEMORY_MANAGEMENT = "memory_management"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    UNKNOWN = "unknown"


class DeveloperErrorCode(str, Enum):
    """Developer error codes for technical debugging."""

    # Code execution errors
    FUNCTION_CALL_FAILED = "DEV_CODE_001"
    INVALID_ARGUMENT_TYPE = "DEV_CODE_002"
    MISSING_REQUIRED_PARAMETER = "DEV_CODE_003"
    RETURN_TYPE_MISMATCH = "DEV_CODE_004"
    LOOP_ITERATION_ERROR = "DEV_CODE_005"

    # Memory management errors
    MEMORY_LEAK_DETECTED = "DEV_MEM_001"
    BUFFER_OVERFLOW = "DEV_MEM_002"
    NULL_POINTER_ACCESS = "DEV_MEM_003"
    GARBAGE_COLLECTION_ERROR = "DEV_MEM_004"

    # Performance errors
    TIMEOUT_EXCEEDED = "DEV_PERF_001"
    RESOURCE_EXHAUSTION = "DEV_PERF_002"
    SLOW_QUERY_DETECTED = "DEV_PERF_003"
    CONCURRENCY_ISSUE = "DEV_PERF_004"

    # Integration errors
    API_CALL_FAILED = "DEV_INT_001"
    SERVICE_UNAVAILABLE = "DEV_INT_002"
    PROTOCOL_MISMATCH = "DEV_INT_003"
    DATA_FORMAT_ERROR = "DEV_INT_004"

    # Dependency errors
    MISSING_DEPENDENCY = "DEV_DEP_001"
    VERSION_CONFLICT = "DEV_DEP_002"
    IMPORT_ERROR = "DEV_DEP_003"
    CIRCULAR_IMPORT = "DEV_DEP_004"

    # Configuration errors
    INVALID_CONFIG_VALUE = "DEV_CONFIG_001"
    MISSING_CONFIG_FILE = "DEV_CONFIG_002"
    ENV_VAR_NOT_SET = "DEV_CONFIG_003"
    CONFIG_PARSE_ERROR = "DEV_CONFIG_004"

    # Network errors
    CONNECTION_REFUSED = "DEV_NET_001"
    DNS_LOOKUP_FAILED = "DEV_NET_002"
    SSL_HANDSHAKE_ERROR = "DEV_NET_003"
    PROXY_AUTHENTICATION_FAILED = "DEV_NET_004"

    # Database errors
    CONNECTION_POOL_EXHAUSTED = "DEV_DB_001"
    QUERY_TIMEOUT = "DEV_DB_002"
    DEADLOCK_DETECTED = "DEV_DB_003"
    SCHEMA_MISMATCH = "DEV_DB_004"

    # Security errors
    AUTHENTICATION_FAILED = "DEV_SEC_001"
    AUTHORIZATION_DENIED = "DEV_SEC_002"
    TOKEN_EXPIRED = "DEV_SEC_003"
    INVALID_SIGNATURE = "DEV_SEC_004"


class StackFrame(BaseModel):
    """Model for stack frame information."""

    filename: str = Field(..., description="Source file name")
    line_number: int = Field(..., description="Line number in the file")
    function_name: str = Field(..., description="Function name")
    code_context: Optional[str] = Field(
        None, description="Code context around the line"
    )
    local_variables: Optional[Dict[str, Any]] = Field(
        None, description="Local variables at the time of error"
    )


class DeveloperErrorContext(BaseModel):
    """Model for developer error context and debugging information."""

    operation: str = Field(..., description="Operation that failed")
    component: str = Field(..., description="Component where error occurred")
    function_name: str = Field(..., description="Function where error occurred")
    timestamp: str = Field(..., description="ISO timestamp of error")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_id: Optional[str] = Field(None, description="Request identifier")

    # Technical debugging information
    stack_trace: List[StackFrame] = Field(
        default_factory=list, description="Stack trace frames"
    )
    local_variables: Optional[Dict[str, Any]] = Field(
        None, description="Local variables at error time"
    )
    global_variables: Optional[Dict[str, Any]] = Field(
        None, description="Global variables at error time"
    )
    function_arguments: Optional[Dict[str, Any]] = Field(
        None, description="Function arguments at error time"
    )

    # System information
    python_version: Optional[str] = Field(None, description="Python version")
    platform_info: Optional[str] = Field(None, description="Platform information")
    memory_usage: Optional[Dict[str, Any]] = Field(
        None, description="Memory usage at error time"
    )
    cpu_usage: Optional[Dict[str, Any]] = Field(
        None, description="CPU usage at error time"
    )

    # Additional context
    user_input: Optional[Dict[str, Any]] = Field(
        None, description="User input that caused error"
    )
    environment: Optional[Dict[str, str]] = Field(
        None, description="Environment variables"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DeveloperError(BaseModel):
    """Comprehensive developer error model with technical details."""

    error_code: DeveloperErrorCode = Field(..., description="Developer error code")
    category: DeveloperErrorCategory = Field(..., description="Error category")
    severity: DeveloperErrorSeverity = Field(..., description="Error severity level")
    title: str = Field(..., description="Technical error title")
    message: str = Field(..., description="Technical error message")

    # Technical error details
    exception_type: str = Field(..., description="Type of exception")
    exception_message: str = Field(..., description="Original exception message")
    stack_trace: str = Field(..., description="Full stack trace")
    stack_frames: List[StackFrame] = Field(
        default_factory=list, description="Parsed stack frames"
    )

    # Context and state data
    context: DeveloperErrorContext = Field(..., description="Error context")

    # Debugging information
    debug_info: Optional[Dict[str, Any]] = Field(
        None, description="Additional debugging information"
    )
    related_errors: List[str] = Field(
        default_factory=list, description="Related error IDs"
    )

    # Metadata
    error_id: Optional[str] = Field(None, description="Unique error identifier")
    created_at: str = Field(..., description="Error creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")

    class Config:
        use_enum_values = True


class DeveloperErrorReport(BaseModel):
    """Model for developer error reporting and aggregation."""

    errors: List[DeveloperError] = Field(
        default_factory=list, description="List of developer errors"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Error summary statistics"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    developer_id: Optional[str] = Field(None, description="Developer identifier")
    environment: Optional[str] = Field(
        None, description="Environment (dev, staging, prod)"
    )
    created_at: str = Field(..., description="Report creation timestamp")

"""Specialized output formatters for different operations.

This module provides formatters for ingestion progress, Q&A sessions,
performance metrics, and error handling with support for different output formats.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..logging import UserOutput
from ..logging.enums import OutputFormat


class IngestionFormatter:
    """Formatter for ingestion operation output."""

    def __init__(self, user_output: UserOutput):
        """Initialize ingestion formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def start_ingestion(self, repo_url: str, settings: Dict[str, Any]) -> None:
        """Format ingestion start message.

        Args:
            repo_url: Repository URL being ingested
            settings: Ingestion settings
        """
        self.user_output.info("🚀 Starting ingestion", emoji="🚀")

        if self.user_output.config.is_verbose():
            self.user_output.verbose(f"Repository: {repo_url}")
            if settings:
                self.user_output.verbose("Settings:")
                for key, value in settings.items():
                    self.user_output.verbose(f"  {key}: {value}")

    def file_processing_progress(self, current: int, total: int, filename: str) -> None:
        """Format file processing progress.

        Args:
            current: Current file number
            total: Total number of files
            filename: Current file being processed
        """
        if self.user_output.config.output_format == OutputFormat.JSON:
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "file_progress",
                "current": current,
                "total": total,
                "filename": filename,
                "percentage": round((current / total * 100), 1) if total > 0 else 0,
            }
            self.user_output.console.print(json.dumps(progress_data))
        else:
            self.user_output.verbose(f"Processing file {current}/{total}: {filename}")

    def embedding_progress(self, current: int, total: int) -> None:
        """Format embedding progress.

        Args:
            current: Current chunk number
            total: Total number of chunks
        """
        if self.user_output.config.output_format == OutputFormat.JSON:
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "embedding_progress",
                "current": current,
                "total": total,
                "percentage": round((current / total * 100), 1) if total > 0 else 0,
            }
            self.user_output.console.print(json.dumps(progress_data))
        else:
            self.user_output.verbose(f"Embedding chunks: {current}/{total}")

    def ingestion_complete(
        self,
        vectorstore_name: str,
        duration: float,
        total_files: int,
        total_chunks: int,
    ) -> None:
        """Format ingestion completion message.

        Args:
            vectorstore_name: Name of the vector store collection
            duration: Total ingestion duration
            total_files: Total number of files processed
            total_chunks: Total number of chunks created
        """
        self.user_output.success("✅ Ingestion completed successfully!", emoji="✅")
        self.user_output.info(f"📚 Collection: {vectorstore_name}")

        if self.user_output.config.is_verbose():
            self.user_output.verbose(f"Duration: {duration:.2f} seconds")
            self.user_output.verbose(f"Files processed: {total_files}")
            self.user_output.verbose(f"Chunks created: {total_chunks}")


class QAFormatter:
    """Formatter for Q&A session output."""

    def __init__(self, user_output: UserOutput):
        """Initialize Q&A formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def session_start(self, repo_id: str) -> None:
        """Format session start message.

        Args:
            repo_id: Repository ID
        """
        self.user_output.info(
            "🤖 Starting enhanced interactive Q&A session...", emoji="🤖"
        )
        self.user_output.info(f"📚 Repository: {repo_id}")

    def question_received(self, question: str) -> None:
        """Format question received message.

        Args:
            question: User question
        """
        if self.user_output.config.output_format == OutputFormat.JSON:
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "question_received",
                "question": question,
            }
            self.user_output.console.print(json.dumps(data))
        else:
            self.user_output.info("🤔 Processing question...", emoji="🤔")

    def answer_generated(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Format answer generation.

        Args:
            question: User question
            answer: Generated answer
            citations: Optional citations
            metadata: Optional metadata
        """
        self.user_output.print_qa_session(question, answer, citations, metadata)

    def session_summary(
        self, duration: float, total_exchanges: int, repo_id: str
    ) -> None:
        """Format session summary.

        Args:
            duration: Session duration
            total_exchanges: Total number of exchanges
            repo_id: Repository ID
        """
        self.user_output.info("📊 Session Summary", emoji="📊")
        self.user_output.info(f"   Duration: {duration:.2f} seconds")
        self.user_output.info(f"   Total exchanges: {total_exchanges}")
        self.user_output.info(f"   Repository: {repo_id}")


class PerformanceFormatter:
    """Formatter for performance metrics output."""

    def __init__(self, user_output: UserOutput):
        """Initialize performance formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def operation_start(self, operation: str) -> None:
        """Format operation start.

        Args:
            operation: Operation name
        """
        if self.user_output.config.output_format == OutputFormat.JSON:
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_start",
                "operation": operation,
            }
            self.user_output.console.print(json.dumps(data))
        else:
            self.user_output.verbose(f"Starting operation: {operation}")

    def operation_complete(
        self,
        operation: str,
        duration: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Format operation completion.

        Args:
            operation: Operation name
            duration: Operation duration
            additional_metrics: Additional performance metrics
        """
        if self.user_output.config.output_format == OutputFormat.JSON:
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_complete",
                "operation": operation,
                "duration": duration,
            }
            if additional_metrics:
                data["metrics"] = additional_metrics
            self.user_output.console.print(json.dumps(data))
        else:
            self.user_output.verbose(
                f"Operation '{operation}' completed in {duration:.2f}s"
            )
            if additional_metrics:
                for key, value in additional_metrics.items():
                    self.user_output.verbose(f"  {key}: {value}")

    def performance_summary(self, metrics: Dict[str, Any]) -> None:
        """Format performance summary.

        Args:
            metrics: Performance metrics dictionary
        """
        self.user_output.print_performance_metrics(metrics)


class ErrorFormatter:
    """Formatter for error output."""

    def __init__(self, user_output: UserOutput):
        """Initialize error formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def operation_error(
        self, operation: str, error: Exception, context: Optional[str] = None
    ) -> None:
        """Format operation error.

        Args:
            operation: Operation that failed
            error: Exception that occurred
            context: Additional context
        """
        self.user_output.error(
            f"❌ {operation} failed", error=error, context=context, emoji="❌"
        )

    def validation_error(self, field: str, value: Any, message: str) -> None:
        """Format validation error.

        Args:
            field: Field that failed validation
            value: Invalid value
            message: Validation message
        """
        if self.user_output.config.output_format == OutputFormat.JSON:
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "validation_error",
                "field": field,
                "value": str(value),
                "message": message,
            }
            self.user_output.console.print(json.dumps(data))
        else:
            self.user_output.error(
                f"Validation error in field '{field}': {message}", emoji="⚠️"
            )

    def network_error(self, url: str, error: Exception) -> None:
        """Format network error.

        Args:
            url: URL that failed
            error: Network error
        """
        self.user_output.error(
            f"Network error accessing {url}",
            error=error,
            context="Network operation",
            emoji="🌐",
        )

    def repository_error(self, repo_url: str, error: Exception) -> None:
        """Format repository-specific error.

        Args:
            repo_url: Repository URL
            error: Repository error
        """
        self.user_output.error(
            f"Repository error for {repo_url}",
            error=error,
            context="Repository operation",
            emoji="📚",
        )

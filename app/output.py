"""Enhanced output formatting for readme-mentor.

This module provides specialized output formatters for different types of
content including ingestion progress, Q&A sessions, performance metrics,
and error handling with support for Rich, Plain, and JSON formats.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from .logging.config import LoggingConfig
from .logging.enums import OutputFormat


class OutputManager:
    """Centralized output management with format-specific formatters."""

    def __init__(self, config: LoggingConfig):
        """Initialize output manager with configuration.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.console = Console(
            file=sys.stdout,
            color_system="auto" if config.should_use_color() else None,
            highlight=False,
        )

    def format_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress information.

        Args:
            current_file: Current file being processed
            current: Current file number
            total: Total number of files
            total_chunks: Total chunks created so far
            duration: Elapsed time

        Returns:
            Formatted progress string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_ingestion_progress(
                current_file, current, total, total_chunks, duration
            )
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_ingestion_progress(
                current_file, current, total, total_chunks, duration
            )
        else:
            return self._format_plain_ingestion_progress(
                current_file, current, total, total_chunks, duration
            )

    def format_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a Q&A response.

        Args:
            question: User question
            answer: AI answer
            citations: Optional citations
            metadata: Optional metadata

        Returns:
            Formatted Q&A response string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_qa_response(question, answer, citations, metadata)
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_qa_response(question, answer, citations, metadata)
        else:
            return self._format_plain_qa_response(question, answer, citations, metadata)

    def format_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary.

        Args:
            metrics: Performance metrics

        Returns:
            Formatted performance summary string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_performance_summary(metrics)
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_performance_summary(metrics)
        else:
            return self._format_plain_performance_summary(metrics)

    def format_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format an error report.

        Args:
            error: The exception
            context: Error context
            suggestions: Optional suggestions

        Returns:
            Formatted error report string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_error_report(error, context, suggestions)
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_error_report(error, context, suggestions)
        else:
            return self._format_plain_error_report(error, context, suggestions)

    def _format_rich_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress with Rich styling."""
        percentage = (current / total * 100) if total > 0 else 0

        content = f"""
[bold blue]Processing:[/bold blue] {current_file}
[bold green]Progress:[/bold green] {current}/{total} files ({percentage:.1f}%)
[bold yellow]Chunks:[/bold yellow] {total_chunks} created
[bold magenta]Duration:[/bold magenta] {duration:.1f}s
"""

        panel = Panel(
            content, title="📄 File Processing", border_style="blue", padding=(1, 2)
        )
        return str(panel)

    def _format_plain_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress as plain text."""
        percentage = (current / total * 100) if total > 0 else 0

        lines = [
            f"Processing: {current_file}",
            f"Progress: {current}/{total} files ({percentage:.1f}%)",
            f"Chunks: {total_chunks} created",
            f"Duration: {duration:.1f}s",
        ]

        return "\n".join(lines)

    def _format_json_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "ingestion_progress",
            "current_file": current_file,
            "current": current,
            "total": total,
            "percentage": round((current / total * 100) if total > 0 else 0, 1),
            "total_chunks": total_chunks,
            "duration_seconds": round(duration, 1),
        }

        return json.dumps(data, indent=2)

    def _format_rich_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format Q&A response with Rich styling."""
        # Question panel
        question_panel = Panel(
            f"[bold blue]❓ Question:[/bold blue]\n{question}",
            title="Question",
            border_style="blue",
        )

        # Answer panel
        answer_panel = Panel(
            f"[bold green]🤖 Answer:[/bold green]\n{answer}",
            title="Answer",
            border_style="green",
        )

        result = str(question_panel) + "\n" + str(answer_panel)

        # Citations table
        if citations:
            citation_table = Table(
                title="📖 Sources", show_header=True, header_style="bold magenta"
            )
            citation_table.add_column("File", style="cyan")
            citation_table.add_column("Lines", style="yellow")
            citation_table.add_column("Content", style="white")

            for citation in citations:
                file_path = citation.get("file", "Unknown")
                start_line = citation.get("start_line", "?")
                end_line = citation.get("end_line", "?")
                content = (
                    citation.get("content", "")[:100] + "..."
                    if len(citation.get("content", "")) > 100
                    else citation.get("content", "")
                )

                citation_table.add_row(file_path, f"{start_line}-{end_line}", content)

            result += "\n" + str(citation_table)

        # Metadata table
        if metadata:
            meta_table = Table(
                title="📊 Metadata", show_header=True, header_style="bold cyan"
            )
            meta_table.add_column("Metric", style="cyan")
            meta_table.add_column("Value", style="green")

            for key, value in metadata.items():
                meta_table.add_row(key.replace("_", " ").title(), str(value))

            result += "\n" + str(meta_table)

        return result

    def _format_plain_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format Q&A response as plain text."""
        lines = ["❓ Question:", question, "", "🤖 Answer:", answer]

        if citations:
            lines.extend(["", "📖 Sources:"])
            for i, citation in enumerate(citations, 1):
                file_path = citation.get("file", "Unknown")
                start_line = citation.get("start_line", "?")
                end_line = citation.get("end_line", "?")
                lines.append(f"  {i}. {file_path} (lines {start_line}-{end_line})")

        if metadata:
            lines.extend(["", "📊 Metadata:"])
            for key, value in metadata.items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)

    def _format_json_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format Q&A response as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "qa_response",
            "question": question,
            "answer": answer,
        }

        if citations:
            data["citations"] = citations

        if metadata:
            data["metadata"] = metadata

        return json.dumps(data, indent=2)

    def _format_rich_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary with Rich styling."""
        table = Table(
            title="Performance Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="dim")

        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    unit = self._get_metric_unit(sub_metric)
                    table.add_row(f"{metric}.{sub_metric}", str(sub_value), unit)
            else:
                unit = self._get_metric_unit(metric)
                table.add_row(metric, str(value), unit)

        return str(table)

    def _format_plain_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary as plain text."""
        lines = ["Performance Summary:"]

        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    unit = self._get_metric_unit(sub_metric)
                    lines.append(f"  {metric}.{sub_metric}: {sub_value} {unit}")
            else:
                unit = self._get_metric_unit(metric)
                lines.append(f"  {metric}: {value} {unit}")

        return "\n".join(lines)

    def _format_json_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance_summary",
            "metrics": metrics,
        }

        return json.dumps(data, indent=2)

    def _format_rich_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format error report with Rich styling."""
        error_content = f"""
[bold red]Error Type:[/bold red] {type(error).__name__}
[bold red]Error Message:[/bold red] {str(error)}
[bold yellow]Context:[/bold yellow] {context}
"""

        if suggestions:
            error_content += "\n[bold cyan]Suggestions:[/bold cyan]\n"
            for suggestion in suggestions:
                error_content += f"• {suggestion}\n"

        panel = Panel(
            error_content, title="❌ Error Report", border_style="red", padding=(1, 2)
        )
        return str(panel)

    def _format_plain_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format error report as plain text."""
        lines = [
            "❌ Error Report:",
            f"  Error Type: {type(error).__name__}",
            f"  Error Message: {str(error)}",
            f"  Context: {context}",
        ]

        if suggestions:
            lines.extend(["", "  Suggestions:"])
            for suggestion in suggestions:
                lines.append(f"    • {suggestion}")

        return "\n".join(lines)

    def _format_json_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format error report as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error_report",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "error_code": self._get_error_code(error),
        }

        if suggestions:
            data["suggestions"] = suggestions

        return json.dumps(data, indent=2)

    def _get_metric_unit(self, metric: str) -> str:
        """Get the unit for a metric."""
        units = {
            "wall_time": "seconds",
            "token_count": "tokens",
            "tool_calls": "calls",
            "memory_usage": "MB",
            "cpu_usage": "%",
            "duration": "seconds",
            "latency": "ms",
        }
        return units.get(metric, "")

    def _get_error_code(self, error: Exception) -> str:
        """Get a standardized error code for an exception."""
        error_message = str(error).lower()

        if "network" in error_message or "connection" in error_message:
            return "NETWORK_ERROR"
        elif "not found" in error_message or "404" in error_message:
            return "NOT_FOUND"
        elif "rate limit" in error_message:
            return "RATE_LIMIT"
        elif "authentication" in error_message or "unauthorized" in error_message:
            return "AUTH_ERROR"
        elif "validation" in error_message:
            return "VALIDATION_ERROR"
        elif "permission" in error_message or "access" in error_message:
            return "PERMISSION_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def create_progress_bar(self, description: str = "") -> Progress:
        """Create a Rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )

    def print(self, content: str) -> None:
        """Print content using the configured format."""
        if not self.config.is_quiet():
            self.console.print(content)

"""Specialized output formatters for different operations.

This module provides formatters for ingestion progress, Q&A sessions,
performance metrics, and error handling with support for different output formats.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.table import Table

from ..logging import UserOutput
from ..logging.enums import OutputFormat


class IngestionFormatter:
    """Enhanced formatter for ingestion operation output with progress indicators."""

    def __init__(self, user_output: UserOutput):
        """Initialize ingestion formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output
        self._progress_bar: Optional[Progress] = None
        self._status_spinner: Optional[Status] = None

    def start_ingestion(self, repo_url: str, settings: Dict[str, Any]) -> None:
        """Format ingestion start message with enhanced status indicators.

        Args:
            repo_url: Repository URL being ingested
            settings: Ingestion settings
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            # Rich format with animated status
            self.user_output.console.print(
                Panel(
                    f"[bold green]🚀 Starting ingestion[/bold green]\n"
                    f"[blue]Repository:[/blue] {repo_url}",
                    title="[bold blue]Ingestion Started[/bold blue]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            if self.user_output.config.is_verbose() and settings:
                settings_table = Table(
                    title="Settings", show_header=True, header_style="bold cyan"
                )
                settings_table.add_column("Setting", style="cyan")
                settings_table.add_column("Value", style="green")

                for key, value in settings.items():
                    settings_table.add_row(key, str(value))

                self.user_output.console.print(settings_table)
        elif self.user_output.config.output_format == OutputFormat.JSON:
            # JSON format with metadata
            start_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "ingestion_start",
                "repository_url": repo_url,
                "settings": settings,
                "status": "started",
            }
            self.user_output.console.print(json.dumps(start_data))
        else:
            # Plain format with structured layout
            self.user_output.info("🚀 Starting ingestion")
            self.user_output.info(f"Repository: {repo_url}")

            if self.user_output.config.is_verbose() and settings:
                self.user_output.info("Settings:")
                for key, value in settings.items():
                    self.user_output.info(f"  {key}: {value}")

    def create_progress_bar(
        self, total_files: int, description: str = "Processing files"
    ) -> Progress:
        """Create an enhanced progress bar for file processing.

        Args:
            total_files: Total number of files to process
            description: Description for the progress bar

        Returns:
            Rich Progress instance
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            self._progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.user_output.console,
                expand=True,
            )
            self._progress_bar.add_task(description, total=total_files)
            return self._progress_bar
        return None

    def update_progress(self, current: int, total: int, filename: str) -> None:
        """Update progress with enhanced formatting.

        Args:
            current: Current file number
            total: Total number of files
            filename: Current file being processed
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            if self._progress_bar:
                self._progress_bar.update(
                    0, completed=current, description=f"Processing: {filename}"
                )
        elif self.user_output.config.output_format == OutputFormat.JSON:
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "file_progress",
                "current": current,
                "total": total,
                "filename": filename,
                "percentage": round((current / total * 100), 1) if total > 0 else 0,
                "status": "processing",
            }
            self.user_output.console.print(json.dumps(progress_data))
        else:
            # Plain format with structured layout
            percentage = (current / total * 100) if total > 0 else 0
            self.user_output.verbose(
                f"Processing file {current}/{total} ({percentage:.1f}%): {filename}"
            )

    def file_processing_progress(self, current: int, total: int, filename: str) -> None:
        """Format file processing progress with enhanced indicators.

        Args:
            current: Current file number
            total: Total number of files
            filename: Current file being processed
        """
        self.update_progress(current, total, filename)

    def embedding_progress(self, current: int, total: int) -> None:
        """Format embedding progress with enhanced indicators.

        Args:
            current: Current chunk number
            total: Total number of chunks
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            percentage = (current / total * 100) if total > 0 else 0
            self.user_output.console.print(
                f"[bold blue]Embedding chunks:[/bold blue] {current}/{total} "
                f"[green]({percentage:.1f}%)[/green]"
            )
        elif self.user_output.config.output_format == OutputFormat.JSON:
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "embedding_progress",
                "current": current,
                "total": total,
                "percentage": round((current / total * 100), 1) if total > 0 else 0,
                "status": "embedding",
            }
            self.user_output.console.print(json.dumps(progress_data))
        else:
            # Plain format with structured layout
            percentage = (current / total * 100) if total > 0 else 0
            self.user_output.verbose(
                f"Embedding chunks: {current}/{total} ({percentage:.1f}%)"
            )

    def ingestion_complete(
        self,
        vectorstore_name: str,
        duration: float,
        total_files: int,
        total_chunks: int,
    ) -> None:
        """Format ingestion completion with enhanced success indicators.

        Args:
            vectorstore_name: Name of the vector store collection
            duration: Total ingestion duration
            total_files: Total number of files processed
            total_chunks: Total number of chunks created
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            # Rich format with success panel and metrics table
            success_panel = Panel(
                f"[bold green]✅ Ingestion completed successfully![/bold green]\n"
                f"[blue]Collection:[/blue] {vectorstore_name}",
                title="[bold green]Success[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            self.user_output.console.print(success_panel)

            # Performance metrics table
            metrics_table = Table(
                title="Performance Summary",
                show_header=True,
                header_style="bold magenta",
            )
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            metrics_table.add_column("Unit", style="dim")

            metrics_table.add_row("Duration", f"{duration:.2f}", "seconds")
            metrics_table.add_row("Files Processed", str(total_files), "files")
            metrics_table.add_row("Chunks Created", str(total_chunks), "chunks")
            metrics_table.add_row(
                "Processing Rate", f"{total_files / duration:.2f}", "files/sec"
            )

            self.user_output.console.print(metrics_table)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            # JSON format with comprehensive metadata
            completion_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "ingestion_complete",
                "status": "success",
                "vectorstore_name": vectorstore_name,
                "duration_seconds": round(duration, 2),
                "total_files": total_files,
                "total_chunks": total_chunks,
                "processing_rate_files_per_sec": round(total_files / duration, 2),
                "metadata": {
                    "operation_type": "repository_ingestion",
                    "completion_time": datetime.now().isoformat(),
                },
            }
            self.user_output.console.print(json.dumps(completion_data))
        else:
            # Plain format with structured layout
            self.user_output.success("✅ Ingestion completed successfully!")
            self.user_output.info(f"📚 Collection: {vectorstore_name}")
            self.user_output.info(f"⏱️  Duration: {duration:.2f} seconds")
            self.user_output.info(f"📄 Files processed: {total_files}")
            self.user_output.info(f"🔗 Chunks created: {total_chunks}")
            self.user_output.info(
                f"⚡ Processing rate: {total_files / duration:.2f} files/sec"
            )

    def close_progress(self) -> None:
        """Close the progress bar and clean up resources."""
        if self._progress_bar:
            self._progress_bar.stop()
            self._progress_bar = None


class QAFormatter:
    """Enhanced formatter for Q&A sessions with structured display."""

    def __init__(self, user_output: UserOutput):
        """Initialize QA formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def session_start(self, repo_id: str) -> None:
        """Format Q&A session start with enhanced status indicators.

        Args:
            repo_id: Repository identifier
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            self.user_output.console.print(
                Panel(
                    f"[bold blue]🤖 Starting Q&A session[/bold blue]\n"
                    f"[cyan]Repository:[/cyan] {repo_id}\n"
                    f"[yellow]Type 'help' for commands, 'quit' to exit[/yellow]",
                    title="[bold blue]Q&A Session[/bold blue]",
                    border_style="blue",
                    padding=(1, 2),
                )
            )
        elif self.user_output.config.output_format == OutputFormat.JSON:
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "qa_session_start",
                "repository_id": repo_id,
                "status": "started",
                "metadata": {
                    "session_type": "interactive_qa",
                    "available_commands": ["help", "quit", "clear"],
                },
            }
            self.user_output.console.print(json.dumps(session_data))
        else:
            # Plain format with structured layout
            self.user_output.info("🤖 Starting Q&A session")
            self.user_output.info(f"Repository: {repo_id}")
            self.user_output.info("Type 'help' for commands, 'quit' to exit")

    def question_received(self, question: str) -> None:
        """Format received question with enhanced display.

        Args:
            question: User question
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            self.user_output.console.print(
                Panel(
                    f"[bold blue]❓ Question:[/bold blue]\n{question}",
                    title="[bold blue]User Input[/bold blue]",
                    border_style="blue",
                    padding=(1, 2),
                )
            )
        elif self.user_output.config.output_format == OutputFormat.JSON:
            question_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "question_received",
                "question": question,
                "status": "processing",
            }
            self.user_output.console.print(json.dumps(question_data))
        else:
            # Plain format with structured layout
            self.user_output.info("❓ Question:")
            self.user_output.info(f"  {question}")

    def answer_generated(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Format generated answer with enhanced structured display.

        Args:
            question: User question
            answer: AI answer
            citations: Optional citations
            metadata: Optional metadata
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            # Answer panel with enhanced styling
            answer_panel = Panel(
                f"[bold green]🤖 Answer:[/bold green]\n{answer}",
                title="[bold green]AI Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            self.user_output.console.print(answer_panel)

            # Citations table if available
            if citations:
                citations_table = Table(
                    title="📖 Sources & Citations",
                    show_header=True,
                    header_style="bold magenta",
                    expand=True,
                )
                citations_table.add_column("File", style="cyan", no_wrap=True)
                citations_table.add_column("Lines", style="yellow")
                citations_table.add_column("Content", style="white")
                citations_table.add_column("Relevance", style="green")

                for citation in citations:
                    file_path = citation.get("file", "Unknown")
                    start_line = citation.get("start_line", "?")
                    end_line = citation.get("end_line", "?")
                    content = citation.get("content", "")
                    relevance = citation.get("relevance", "N/A")

                    # Truncate content for display
                    display_content = (
                        content[:150] + "..." if len(content) > 150 else content
                    )

                    citations_table.add_row(
                        file_path,
                        f"{start_line}-{end_line}",
                        display_content,
                        str(relevance),
                    )

                self.user_output.console.print(citations_table)

            # Metadata table if available
            if metadata:
                metadata_table = Table(
                    title="📊 Response Metadata",
                    show_header=True,
                    header_style="bold cyan",
                )
                metadata_table.add_column("Metric", style="cyan")
                metadata_table.add_column("Value", style="green")

                for key, value in metadata.items():
                    formatted_key = key.replace("_", " ").title()
                    metadata_table.add_row(formatted_key, str(value))

                self.user_output.console.print(metadata_table)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            # JSON format with comprehensive structure
            answer_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "answer_generated",
                "question": question,
                "answer": answer,
                "status": "completed",
            }

            if citations:
                answer_data["citations"] = citations

            if metadata:
                answer_data["metadata"] = metadata

            self.user_output.console.print(json.dumps(answer_data))
        else:
            # Plain format with structured layout
            self.user_output.info("🤖 Answer:")
            self.user_output.info(f"  {answer}")

            if citations:
                self.user_output.info("📖 Sources:")
                for i, citation in enumerate(citations, 1):
                    file_path = citation.get("file", "Unknown")
                    start_line = citation.get("start_line", "?")
                    end_line = citation.get("end_line", "?")
                    self.user_output.info(
                        f"  {i}. {file_path} (lines {start_line}-{end_line})"
                    )

            if metadata:
                self.user_output.info("📊 Metadata:")
                for key, value in metadata.items():
                    formatted_key = key.replace("_", " ").title()
                    self.user_output.info(f"  {formatted_key}: {value}")

    def session_summary(
        self, duration: float, total_exchanges: int, repo_id: str
    ) -> None:
        """Format session summary with enhanced metrics display.

        Args:
            duration: Session duration
            total_exchanges: Total Q&A exchanges
            repo_id: Repository identifier
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            summary_panel = Panel(
                f"[bold green]✅ Session completed[/bold green]\n"
                f"[blue]Repository:[/blue] {repo_id}\n"
                f"[yellow]Duration:[/yellow] {duration:.2f} seconds\n"
                f"[magenta]Exchanges:[/magenta] {total_exchanges}",
                title="[bold green]Session Summary[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            self.user_output.console.print(summary_panel)

            # Performance metrics table
            if total_exchanges > 0:
                avg_time = duration / total_exchanges
                metrics_table = Table(
                    title="Performance Metrics",
                    show_header=True,
                    header_style="bold magenta",
                )
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="green")
                metrics_table.add_column("Unit", style="dim")

                metrics_table.add_row("Total Duration", f"{duration:.2f}", "seconds")
                metrics_table.add_row(
                    "Total Exchanges", str(total_exchanges), "exchanges"
                )
                metrics_table.add_row(
                    "Average Time per Exchange", f"{avg_time:.2f}", "seconds"
                )
                metrics_table.add_row(
                    "Exchange Rate",
                    f"{total_exchanges / duration:.2f}",
                    "exchanges/sec",
                )

                self.user_output.console.print(metrics_table)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "qa_session_summary",
                "repository_id": repo_id,
                "duration_seconds": round(duration, 2),
                "total_exchanges": total_exchanges,
                "average_time_per_exchange": round(duration / total_exchanges, 2)
                if total_exchanges > 0
                else 0,
                "exchange_rate_per_sec": round(total_exchanges / duration, 2)
                if duration > 0
                else 0,
                "status": "completed",
                "metadata": {
                    "session_type": "interactive_qa",
                    "completion_time": datetime.now().isoformat(),
                },
            }
            self.user_output.console.print(json.dumps(summary_data))
        else:
            # Plain format with structured layout
            self.user_output.success("✅ Session completed")
            self.user_output.info(f"Repository: {repo_id}")
            self.user_output.info(f"⏱️  Duration: {duration:.2f} seconds")
            self.user_output.info(f"💬 Exchanges: {total_exchanges}")

            if total_exchanges > 0:
                avg_time = duration / total_exchanges
                self.user_output.info(
                    f"📊 Average time per exchange: {avg_time:.2f} seconds"
                )
                self.user_output.info(
                    f"⚡ Exchange rate: {total_exchanges / duration:.2f} exchanges/sec"
                )


class PerformanceFormatter:
    """Enhanced formatter for performance metrics with structured display."""

    def __init__(self, user_output: UserOutput):
        """Initialize performance formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def operation_start(self, operation: str) -> None:
        """Format operation start with enhanced status indicators.

        Args:
            operation: Operation name
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            self.user_output.console.print(
                f"[bold blue]🔄 Starting:[/bold blue] {operation}", style="blue"
            )
        elif self.user_output.config.output_format == OutputFormat.JSON:
            start_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_start",
                "operation": operation,
                "status": "started",
            }
            self.user_output.console.print(json.dumps(start_data))
        else:
            # Plain format with structured layout
            self.user_output.info(f"🔄 Starting: {operation}")

    def operation_complete(
        self,
        operation: str,
        duration: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Format operation completion with enhanced metrics display.

        Args:
            operation: Operation name
            duration: Operation duration
            additional_metrics: Additional performance metrics
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            # Success message with enhanced styling
            self.user_output.console.print(
                f"[bold green]✅ Completed:[/bold green] {operation} "
                f"[yellow]({duration:.2f}s)[/yellow]",
                style="green",
            )

            # Additional metrics table if available
            if additional_metrics:
                metrics_table = Table(
                    title=f"Performance Metrics - {operation}",
                    show_header=True,
                    header_style="bold magenta",
                )
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="green")
                metrics_table.add_column("Unit", style="dim")

                for key, value in additional_metrics.items():
                    unit = self._get_metric_unit(key)
                    formatted_key = key.replace("_", " ").title()
                    metrics_table.add_row(formatted_key, str(value), unit)

                self.user_output.console.print(metrics_table)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            completion_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_complete",
                "operation": operation,
                "duration_seconds": round(duration, 2),
                "status": "completed",
            }

            if additional_metrics:
                completion_data["metrics"] = additional_metrics

            self.user_output.console.print(json.dumps(completion_data))
        else:
            # Plain format with structured layout
            self.user_output.success(f"✅ Completed: {operation} ({duration:.2f}s)")

            if additional_metrics:
                self.user_output.info("📊 Additional metrics:")
                for key, value in additional_metrics.items():
                    unit = self._get_metric_unit(key)
                    formatted_key = key.replace("_", " ").title()
                    self.user_output.info(f"  {formatted_key}: {value} {unit}")

    def performance_summary(self, metrics: Dict[str, Any]) -> None:
        """Format performance summary with enhanced structured display.

        Args:
            metrics: Performance metrics dictionary
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            # Comprehensive performance summary table
            summary_table = Table(
                title="Performance Summary",
                show_header=True,
                header_style="bold magenta",
                expand=True,
            )
            summary_table.add_column("Metric", style="cyan", no_wrap=True)
            summary_table.add_column("Value", style="green")
            summary_table.add_column("Unit", style="dim")
            summary_table.add_column("Category", style="yellow")

            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        unit = self._get_metric_unit(sub_metric)
                        category = metric.replace("_", " ").title()
                        summary_table.add_row(
                            f"{metric}.{sub_metric}", str(sub_value), unit, category
                        )
                else:
                    unit = self._get_metric_unit(metric)
                    category = "General"
                    summary_table.add_row(metric, str(value), unit, category)

            self.user_output.console.print(summary_table)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            # JSON format with comprehensive metadata
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "performance_summary",
                "metrics": metrics,
                "metadata": {
                    "summary_type": "comprehensive",
                    "generated_at": datetime.now().isoformat(),
                },
            }
            self.user_output.console.print(json.dumps(summary_data))
        else:
            # Plain format with structured layout
            self.user_output.info("📊 Performance Summary:")

            for metric, value in metrics.items():
                if isinstance(value, dict):
                    self.user_output.info(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        unit = self._get_metric_unit(sub_metric)
                        self.user_output.info(f"    {sub_metric}: {sub_value} {unit}")
                else:
                    unit = self._get_metric_unit(metric)
                    self.user_output.info(f"  {metric}: {value} {unit}")

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
            "throughput": "items/sec",
            "accuracy": "%",
            "precision": "%",
            "recall": "%",
            "f1_score": "%",
        }
        return units.get(metric, "")


class ErrorFormatter:
    """Enhanced formatter for error handling with clear suggestions."""

    def __init__(self, user_output: UserOutput):
        """Initialize error formatter.

        Args:
            user_output: User output instance
        """
        self.user_output = user_output

    def operation_error(
        self, operation: str, error: Exception, context: Optional[str] = None
    ) -> None:
        """Format operation error with enhanced error display and suggestions.

        Args:
            operation: Operation that failed
            error: Exception that occurred
            context: Additional context information
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            # Enhanced error panel with suggestions
            error_content = f"""
[bold red]❌ Operation Failed:[/bold red] {operation}
[bold red]Error Type:[/bold red] {type(error).__name__}
[bold red]Error Message:[/bold red] {str(error)}
"""

            if context:
                error_content += f"[bold yellow]Context:[/bold yellow] {context}\n"

            suggestions = self._get_error_suggestions(error)
            if suggestions:
                error_content += "\n[bold cyan]💡 Suggestions:[/bold cyan]\n"
                for suggestion in suggestions:
                    error_content += f"• {suggestion}\n"

            error_panel = Panel(
                error_content,
                title="[bold red]Error Report[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            self.user_output.console.print(error_panel)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            # JSON format with standardized error structure
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_error",
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_code": self._get_error_code(error),
                "status": "failed",
            }

            if context:
                error_data["context"] = context

            suggestions = self._get_error_suggestions(error)
            if suggestions:
                error_data["suggestions"] = suggestions

            self.user_output.console.print(json.dumps(error_data))
        else:
            # Plain format with structured layout and clear error messages
            self.user_output.error(f"❌ Operation Failed: {operation}")
            self.user_output.error(f"Error Type: {type(error).__name__}")
            self.user_output.error(f"Error Message: {str(error)}")

            if context:
                self.user_output.error(f"Context: {context}")

            suggestions = self._get_error_suggestions(error)
            if suggestions:
                self.user_output.info("💡 Suggestions:")
                for suggestion in suggestions:
                    self.user_output.info(f"  • {suggestion}")

    def validation_error(self, field: str, value: Any, message: str) -> None:
        """Format validation error with enhanced display.

        Args:
            field: Field that failed validation
            value: Invalid value
            message: Validation message
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            validation_panel = Panel(
                f"[bold red]❌ Validation Error[/bold red]\n"
                f"[yellow]Field:[/yellow] {field}\n"
                f"[yellow]Value:[/yellow] {value}\n"
                f"[red]Message:[/red] {message}",
                title="[bold red]Validation Failed[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            self.user_output.console.print(validation_panel)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            validation_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "validation_error",
                "field": field,
                "value": str(value),
                "message": message,
                "error_code": "VALIDATION_ERROR",
                "status": "failed",
            }
            self.user_output.console.print(json.dumps(validation_data))
        else:
            # Plain format with structured layout
            self.user_output.error("❌ Validation Error")
            self.user_output.error(f"Field: {field}")
            self.user_output.error(f"Value: {value}")
            self.user_output.error(f"Message: {message}")

    def network_error(self, url: str, error: Exception) -> None:
        """Format network error with enhanced suggestions.

        Args:
            url: URL that failed
            error: Network error
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            network_panel = Panel(
                f"[bold red]🌐 Network Error[/bold red]\n"
                f"[yellow]URL:[/yellow] {url}\n"
                f"[red]Error:[/red] {str(error)}\n\n"
                f"[bold cyan]💡 Suggestions:[/bold cyan]\n"
                f"• Check your internet connection\n"
                f"• Verify the URL is accessible\n"
                f"• Try again in a few moments\n"
                f"• Check if the service is available",
                title="[bold red]Network Issue[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            self.user_output.console.print(network_panel)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            network_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "network_error",
                "url": url,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_code": "NETWORK_ERROR",
                "status": "failed",
                "suggestions": [
                    "Check your internet connection",
                    "Verify the URL is accessible",
                    "Try again in a few moments",
                    "Check if the service is available",
                ],
            }
            self.user_output.console.print(json.dumps(network_data))
        else:
            # Plain format with structured layout
            self.user_output.error("🌐 Network Error")
            self.user_output.error(f"URL: {url}")
            self.user_output.error(f"Error: {str(error)}")
            self.user_output.info("💡 Suggestions:")
            self.user_output.info("  • Check your internet connection")
            self.user_output.info("  • Verify the URL is accessible")
            self.user_output.info("  • Try again in a few moments")
            self.user_output.info("  • Check if the service is available")

    def repository_error(self, repo_url: str, error: Exception) -> None:
        """Format repository error with enhanced suggestions.

        Args:
            repo_url: Repository URL that failed
            error: Repository error
        """
        if self.user_output.config.output_format == OutputFormat.RICH:
            repo_panel = Panel(
                f"[bold red]📚 Repository Error[/bold red]\n"
                f"[yellow]Repository:[/yellow] {repo_url}\n"
                f"[red]Error:[/red] {str(error)}\n\n"
                f"[bold cyan]💡 Suggestions:[/bold cyan]\n"
                f"• Verify the repository URL is correct\n"
                f"• Check if the repository is public\n"
                f"• Ensure you have access to the repository\n"
                f"• Check your GitHub credentials if private",
                title="[bold red]Repository Issue[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            self.user_output.console.print(repo_panel)

        elif self.user_output.config.output_format == OutputFormat.JSON:
            repo_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "repository_error",
                "repository_url": repo_url,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_code": "REPOSITORY_ERROR",
                "status": "failed",
                "suggestions": [
                    "Verify the repository URL is correct",
                    "Check if the repository is public",
                    "Ensure you have access to the repository",
                    "Check your GitHub credentials if private",
                ],
            }
            self.user_output.console.print(json.dumps(repo_data))
        else:
            # Plain format with structured layout
            self.user_output.error("📚 Repository Error")
            self.user_output.error(f"Repository: {repo_url}")
            self.user_output.error(f"Error: {str(error)}")
            self.user_output.info("💡 Suggestions:")
            self.user_output.info("  • Verify the repository URL is correct")
            self.user_output.info("  • Check if the repository is public")
            self.user_output.info("  • Ensure you have access to the repository")
            self.user_output.info("  • Check your GitHub credentials if private")

    def _get_error_suggestions(self, error: Exception) -> List[str]:
        """Get suggestions based on error type."""
        error_message = str(error).lower()

        if any(word in error_message for word in ["network", "connection", "timeout"]):
            return [
                "Check your internet connection",
                "Verify the service is accessible",
                "Try again in a few moments",
            ]
        elif any(
            word in error_message for word in ["not found", "404", "does not exist"]
        ):
            return [
                "Verify the URL or path is correct",
                "Check if the resource exists",
                "Ensure you have proper access permissions",
            ]
        elif any(word in error_message for word in ["rate limit", "too many requests"]):
            return [
                "Wait a few minutes before trying again",
                "Check your API rate limits",
                "Consider using authentication for higher limits",
            ]
        elif any(
            word in error_message
            for word in ["authentication", "unauthorized", "forbidden"]
        ):
            return [
                "Check your credentials",
                "Verify your access permissions",
                "Ensure your token is valid and not expired",
            ]
        elif any(
            word in error_message for word in ["validation", "invalid", "malformed"]
        ):
            return [
                "Check the input format and requirements",
                "Verify all required fields are provided",
                "Ensure the data meets validation rules",
            ]
        elif any(word in error_message for word in ["permission", "access denied"]):
            return [
                "Check file and directory permissions",
                "Ensure you have write access to the target directory",
                "Try running with elevated privileges if needed",
            ]
        else:
            return [
                "Check the error message for specific details",
                "Verify your input and configuration",
                "Try the operation again",
                "Contact support if the issue persists",
            ]

    def _get_error_code(self, error: Exception) -> str:
        """Get a standardized error code for an exception."""
        error_message = str(error).lower()

        if any(word in error_message for word in ["network", "connection", "timeout"]):
            return "NETWORK_ERROR"
        elif any(
            word in error_message for word in ["not found", "404", "does not exist"]
        ):
            return "NOT_FOUND"
        elif any(word in error_message for word in ["rate limit", "too many requests"]):
            return "RATE_LIMIT"
        elif any(
            word in error_message
            for word in ["authentication", "unauthorized", "forbidden"]
        ):
            return "AUTH_ERROR"
        elif any(
            word in error_message for word in ["validation", "invalid", "malformed"]
        ):
            return "VALIDATION_ERROR"
        elif any(word in error_message for word in ["permission", "access denied"]):
            return "PERMISSION_ERROR"
        else:
            return "UNKNOWN_ERROR"

"""Metrics Display Implementation for Phase 5.2.

This module provides comprehensive metrics display functionality with support for:
- Normal Mode: Key metrics summary
- Verbose Mode: Detailed breakdown
- Debug Mode: Complete analysis with trends
- JSON Mode: Structured data for analysis
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..logging.enums import OutputFormat, VerbosityLevel
from .models import MetricsData, OperationMetrics


class MetricsDisplayFormatter:
    """Comprehensive metrics display formatter with multiple verbosity levels."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the metrics display formatter.

        Args:
            console: Rich console instance for output
        """
        self.console = console or Console()

    def display_metrics(
        self,
        metrics_data: MetricsData,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        output_format: OutputFormat = OutputFormat.RICH,
    ) -> None:
        """Display metrics data based on verbosity level and output format.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for display detail
            output_format: Output format (rich, plain, json)
        """
        if output_format == OutputFormat.JSON:
            self._display_json(metrics_data, verbosity)
        elif output_format == OutputFormat.RICH:
            self._display_rich(metrics_data, verbosity)
        else:
            self._display_plain(metrics_data, verbosity)

    def _display_json(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display metrics in JSON format.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        json_data = self._format_metrics_as_json(metrics_data, verbosity)
        self.console.print(json.dumps(json_data, indent=2, default=str))

    def _display_rich(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display metrics using Rich console formatting.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        # Display session summary
        self._display_session_summary_rich(metrics_data, verbosity)

        # Display operations based on verbosity
        if verbosity >= VerbosityLevel.NORMAL:
            self._display_operations_summary_rich(metrics_data, verbosity)

        if verbosity >= VerbosityLevel.VERBOSE:
            self._display_detailed_operations_rich(metrics_data)

        if verbosity >= VerbosityLevel.DEBUG:
            self._display_debug_analysis_rich(metrics_data)

    def _display_plain(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display metrics in plain text format.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        # Display session summary
        self._display_session_summary_plain(metrics_data, verbosity)

        # Display operations based on verbosity
        if verbosity >= VerbosityLevel.NORMAL:
            self._display_operations_summary_plain(metrics_data, verbosity)

        if verbosity >= VerbosityLevel.VERBOSE:
            self._display_detailed_operations_plain(metrics_data)

        if verbosity >= VerbosityLevel.DEBUG:
            self._display_debug_analysis_plain(metrics_data)

    def _display_session_summary_rich(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display session summary using Rich formatting.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        # Create summary panel
        summary_text = f"""
Session ID: {metrics_data.session_id}
Start Time: {metrics_data.start_time.strftime("%Y-%m-%d %H:%M:%S")}
Total Operations: {metrics_data.total_operations}
Successful: {metrics_data.successful_operations}
Failed: {metrics_data.failed_operations}
Success Rate: {(metrics_data.successful_operations / max(metrics_data.total_operations, 1) * 100):.1f}%
        """.strip()

        if verbosity >= VerbosityLevel.NORMAL:
            summary_text += f"""
Total Duration: {metrics_data.total_duration:.3f}s
Avg Duration: {metrics_data.avg_operation_duration:.3f}s
Total Tokens: {metrics_data.total_tokens:,}
Total Cost: ${metrics_data.total_cost:.4f}
            """.strip()

        if verbosity >= VerbosityLevel.VERBOSE:
            summary_text += f"""
Tool Calls: {metrics_data.total_tool_calls}
Successful Tool Calls: {metrics_data.successful_tool_calls}
Failed Tool Calls: {metrics_data.failed_tool_calls}
Tool Call Success Rate: {(metrics_data.successful_tool_calls / max(metrics_data.total_tool_calls, 1) * 100):.1f}%
            """.strip()

        # Determine panel color based on success rate
        success_rate = metrics_data.successful_operations / max(
            metrics_data.total_operations, 1
        )
        if success_rate >= 0.9:
            border_color = "green"
        elif success_rate >= 0.7:
            border_color = "yellow"
        else:
            border_color = "red"

        self.console.print(
            Panel(
                summary_text,
                title="[bold blue]Session Summary[/bold blue]",
                border_style=border_color,
                padding=(1, 2),
            )
        )

    def _display_operations_summary_rich(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display operations summary using Rich formatting.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        if not metrics_data.operations:
            return

        # Create operations table
        table = Table(
            title="[bold blue]Operations Summary[/bold blue]",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Operation", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Tokens", style="magenta")

        if verbosity >= VerbosityLevel.VERBOSE:
            table.add_column("Cost", style="red")
            table.add_column("Tool Calls", style="blue")

        # Group operations by type
        operation_groups: Dict[str, List[OperationMetrics]] = {}
        for op in metrics_data.operations:
            op_type = op.operation_name
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(op)

        # Add rows for each operation type
        for op_type, operations in operation_groups.items():
            total_ops = len(operations)
            successful_ops = sum(1 for op in operations if op.success)
            avg_duration = sum(op.total_duration or 0 for op in operations) / total_ops
            total_tokens = sum(
                op.token_usage.total_tokens if op.token_usage else 0
                for op in operations
            )
            total_cost = sum(
                op.cost_estimate.total_cost if op.cost_estimate else 0
                for op in operations
            )
            total_tool_calls = sum(len(op.tool_calls) for op in operations)

            # Status indicator
            success_rate = successful_ops / total_ops
            if success_rate >= 0.9:
                status = "✅"
            elif success_rate >= 0.7:
                status = "⚠️"
            else:
                status = "❌"

            row_data = [
                op_type,
                f"{status} {successful_ops}/{total_ops}",
                f"{avg_duration:.3f}s",
                f"{total_tokens:,}",
            ]

            if verbosity >= VerbosityLevel.VERBOSE:
                row_data.extend(
                    [
                        f"${total_cost:.4f}",
                        f"{total_tool_calls}",
                    ]
                )

            table.add_row(*row_data)

        self.console.print(table)

    def _display_detailed_operations_rich(self, metrics_data: MetricsData) -> None:
        """Display detailed operations breakdown using Rich formatting.

        Args:
            metrics_data: Metrics data to display
        """
        if not metrics_data.operations:
            return

        # Create detailed operations table
        table = Table(
            title="[bold blue]Detailed Operations[/bold blue]",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Operation", style="cyan", no_wrap=True)
        table.add_column("Start Time", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Errors", style="red")
        table.add_column("Component Timing", style="blue")

        for op in metrics_data.operations:
            # Format start time
            start_time = datetime.fromtimestamp(op.start_time).strftime("%H:%M:%S")

            # Status indicator
            status = "✅" if op.success else "❌"

            # Component timing summary
            if op.component_timing:
                timing_summary = ", ".join(
                    f"{comp}: {dur:.3f}s"
                    for comp, dur in sorted(op.component_timing.items())
                )
            else:
                timing_summary = "N/A"

            table.add_row(
                op.operation_name,
                start_time,
                f"{op.total_duration:.3f}s" if op.total_duration else "N/A",
                f"{status} {op.error_count} errors",
                str(op.error_count),
                timing_summary,
            )

        self.console.print(table)

    def _display_debug_analysis_rich(self, metrics_data: MetricsData) -> None:
        """Display debug analysis with trends and optimization suggestions.

        Args:
            metrics_data: Metrics data to display
        """
        # Display performance trends
        if metrics_data.trends:
            self.console.print(
                Panel(
                    "[bold blue]Performance Trends Analysis[/bold blue]",
                    border_style="blue",
                )
            )

            for trend in metrics_data.trends:
                trend_table = Table(
                    title=f"Trend: {trend.operation_type}",
                    show_header=True,
                    header_style="bold cyan",
                )

                trend_table.add_column("Metric", style="cyan")
                trend_table.add_column("Value", style="green")
                trend_table.add_column("Trend", style="yellow")

                # Add trend data
                trend_table.add_row(
                    "Avg Duration",
                    f"{trend.avg_duration:.3f}s",
                    f"{trend.trend_direction} ({trend.trend_strength:.2f})",
                )
                trend_table.add_row(
                    "Success Rate",
                    f"{trend.success_rate:.1f}%",
                    "",
                )
                trend_table.add_row(
                    "Avg Cost",
                    f"${trend.avg_cost:.4f}",
                    "",
                )
                trend_table.add_row(
                    "Avg Tokens",
                    f"{trend.avg_tokens:.0f}",
                    "",
                )

                self.console.print(trend_table)

                # Display bottlenecks and suggestions
                if trend.bottlenecks or trend.optimization_suggestions:
                    suggestions_text = ""
                    if trend.bottlenecks:
                        suggestions_text += (
                            f"Bottlenecks: {', '.join(trend.bottlenecks)}\n"
                        )
                    if trend.optimization_suggestions:
                        suggestions_text += (
                            f"Suggestions: {', '.join(trend.optimization_suggestions)}"
                        )

                    self.console.print(
                        Panel(
                            suggestions_text,
                            title="[bold yellow]Optimization Insights[/bold yellow]",
                            border_style="yellow",
                        )
                    )

        # Display system metrics if available
        if hasattr(metrics_data, "system_metrics"):
            self.console.print(
                Panel(
                    "[bold blue]System Metrics[/bold blue]",
                    border_style="blue",
                )
            )

            sys_table = Table(show_header=True, header_style="bold cyan")
            sys_table.add_column("Metric", style="cyan")
            sys_table.add_column("Value", style="green")

            for key, value in metrics_data.system_metrics.items():
                sys_table.add_row(key, str(value))

            self.console.print(sys_table)

    def _display_session_summary_plain(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display session summary in plain text format.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        self.console.print("=" * 60)
        self.console.print("SESSION SUMMARY")
        self.console.print("=" * 60)
        self.console.print(f"Session ID: {metrics_data.session_id}")
        self.console.print(
            f"Start Time: {metrics_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.console.print(f"Total Operations: {metrics_data.total_operations}")
        self.console.print(f"Successful: {metrics_data.successful_operations}")
        self.console.print(f"Failed: {metrics_data.failed_operations}")

        success_rate = (
            metrics_data.successful_operations
            / max(metrics_data.total_operations, 1)
            * 100
        )
        self.console.print(f"Success Rate: {success_rate:.1f}%")

        if verbosity >= VerbosityLevel.NORMAL:
            self.console.print(f"Total Duration: {metrics_data.total_duration:.3f}s")
            self.console.print(
                f"Avg Duration: {metrics_data.avg_operation_duration:.3f}s"
            )
            self.console.print(f"Total Tokens: {metrics_data.total_tokens:,}")
            self.console.print(f"Total Cost: ${metrics_data.total_cost:.4f}")

        if verbosity >= VerbosityLevel.VERBOSE:
            self.console.print(f"Tool Calls: {metrics_data.total_tool_calls}")
            self.console.print(
                f"Successful Tool Calls: {metrics_data.successful_tool_calls}"
            )
            self.console.print(f"Failed Tool Calls: {metrics_data.failed_tool_calls}")

            tool_success_rate = (
                metrics_data.successful_tool_calls
                / max(metrics_data.total_tool_calls, 1)
                * 100
            )
            self.console.print(f"Tool Call Success Rate: {tool_success_rate:.1f}%")

        self.console.print()

    def _display_operations_summary_plain(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> None:
        """Display operations summary in plain text format.

        Args:
            metrics_data: Metrics data to display
            verbosity: Verbosity level for detail
        """
        if not metrics_data.operations:
            return

        self.console.print("OPERATIONS SUMMARY")
        self.console.print("-" * 40)

        # Group operations by type
        operation_groups: Dict[str, List[OperationMetrics]] = {}
        for op in metrics_data.operations:
            op_type = op.operation_name
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(op)

        for op_type, operations in operation_groups.items():
            total_ops = len(operations)
            successful_ops = sum(1 for op in operations if op.success)
            avg_duration = sum(op.total_duration or 0 for op in operations) / total_ops
            total_tokens = sum(
                op.token_usage.total_tokens if op.token_usage else 0
                for op in operations
            )

            self.console.print(f"{op_type}:")
            self.console.print(f"  Operations: {successful_ops}/{total_ops} successful")
            self.console.print(f"  Avg Duration: {avg_duration:.3f}s")
            self.console.print(f"  Total Tokens: {total_tokens:,}")

            if verbosity >= VerbosityLevel.VERBOSE:
                total_cost = sum(
                    op.cost_estimate.total_cost if op.cost_estimate else 0
                    for op in operations
                )
                total_tool_calls = sum(len(op.tool_calls) for op in operations)
                self.console.print(f"  Total Cost: ${total_cost:.4f}")
                self.console.print(f"  Tool Calls: {total_tool_calls}")

            self.console.print()

    def _display_detailed_operations_plain(self, metrics_data: MetricsData) -> None:
        """Display detailed operations in plain text format.

        Args:
            metrics_data: Metrics data to display
        """
        if not metrics_data.operations:
            return

        self.console.print("DETAILED OPERATIONS")
        self.console.print("-" * 40)

        for op in metrics_data.operations:
            start_time = datetime.fromtimestamp(op.start_time).strftime("%H:%M:%S")
            status = "SUCCESS" if op.success else "FAILED"

            self.console.print(f"{op.operation_name} ({start_time}):")
            self.console.print(f"  Status: {status}")
            self.console.print(
                f"  Duration: {op.total_duration:.3f}s"
                if op.total_duration
                else "  Duration: N/A"
            )
            self.console.print(f"  Errors: {op.error_count}")

            if op.component_timing:
                self.console.print("  Component Timing:")
                for comp, dur in sorted(op.component_timing.items()):
                    self.console.print(f"    {comp}: {dur:.3f}s")

            self.console.print()

    def _display_debug_analysis_plain(self, metrics_data: MetricsData) -> None:
        """Display debug analysis in plain text format.

        Args:
            metrics_data: Metrics data to display
        """
        if metrics_data.trends:
            self.console.print("PERFORMANCE TRENDS")
            self.console.print("-" * 40)

            for trend in metrics_data.trends:
                self.console.print(f"Operation Type: {trend.operation_type}")
                self.console.print(f"  Avg Duration: {trend.avg_duration:.3f}s")
                self.console.print(f"  Success Rate: {trend.success_rate:.1f}%")
                self.console.print(
                    f"  Trend: {trend.trend_direction} (strength: {trend.trend_strength:.2f})"
                )
                self.console.print(f"  Avg Cost: ${trend.avg_cost:.4f}")
                self.console.print(f"  Avg Tokens: {trend.avg_tokens:.0f}")

                if trend.bottlenecks:
                    self.console.print(f"  Bottlenecks: {', '.join(trend.bottlenecks)}")

                if trend.optimization_suggestions:
                    self.console.print(
                        f"  Suggestions: {', '.join(trend.optimization_suggestions)}"
                    )

                self.console.print()

    def _format_metrics_as_json(
        self, metrics_data: MetricsData, verbosity: VerbosityLevel
    ) -> Dict[str, Any]:
        """Format metrics data as JSON structure.

        Args:
            metrics_data: Metrics data to format
            verbosity: Verbosity level for detail

        Returns:
            JSON-compatible dictionary
        """
        json_data = {
            "session_id": metrics_data.session_id,
            "start_time": metrics_data.start_time.isoformat(),
            "end_time": metrics_data.end_time.isoformat()
            if metrics_data.end_time
            else None,
            "summary": {
                "total_operations": metrics_data.total_operations,
                "successful_operations": metrics_data.successful_operations,
                "failed_operations": metrics_data.failed_operations,
                "success_rate": (
                    metrics_data.successful_operations
                    / max(metrics_data.total_operations, 1)
                    * 100
                ),
                "total_duration": metrics_data.total_duration,
                "avg_operation_duration": metrics_data.avg_operation_duration,
                "total_tokens": metrics_data.total_tokens,
                "total_cost": metrics_data.total_cost,
            },
        }

        if verbosity >= VerbosityLevel.VERBOSE:
            json_data["summary"].update(
                {
                    "total_tool_calls": metrics_data.total_tool_calls,
                    "successful_tool_calls": metrics_data.successful_tool_calls,
                    "failed_tool_calls": metrics_data.failed_tool_calls,
                    "tool_call_success_rate": (
                        metrics_data.successful_tool_calls
                        / max(metrics_data.total_tool_calls, 1)
                        * 100
                    ),
                }
            )

        if verbosity >= VerbosityLevel.NORMAL:
            json_data["operations"] = []
            for op in metrics_data.operations:
                op_data = {
                    "operation_name": op.operation_name,
                    "start_time": op.start_time,
                    "end_time": op.end_time,
                    "total_duration": op.total_duration,
                    "success": op.success,
                    "error_count": op.error_count,
                    "warning_count": op.warning_count,
                }

                if verbosity >= VerbosityLevel.VERBOSE:
                    op_data.update(
                        {
                            "component_timing": op.component_timing,
                            "tool_calls": [
                                {
                                    "tool_name": tc.tool_name,
                                    "status": tc.status.value,
                                    "duration": tc.duration,
                                    "error_category": tc.error_category.value
                                    if tc.error_category
                                    else None,
                                    "error_message": tc.error_message,
                                }
                                for tc in op.tool_calls
                            ],
                        }
                    )

                    if op.token_usage:
                        op_data["token_usage"] = {
                            "input_tokens": op.token_usage.input_tokens,
                            "output_tokens": op.token_usage.output_tokens,
                            "total_tokens": op.token_usage.total_tokens,
                            "model_name": op.token_usage.model_name,
                        }

                    if op.cost_estimate:
                        op_data["cost_estimate"] = {
                            "total_cost": op.cost_estimate.total_cost,
                            "input_cost": op.cost_estimate.input_cost,
                            "output_cost": op.cost_estimate.output_cost,
                            "currency": op.cost_estimate.currency,
                            "model_name": op.cost_estimate.model_name,
                        }

                json_data["operations"].append(op_data)

        if verbosity >= VerbosityLevel.DEBUG and metrics_data.trends:
            json_data["trends"] = [
                {
                    "operation_type": trend.operation_type,
                    "time_period": trend.time_period,
                    "data_points": trend.data_points,
                    "avg_duration": trend.avg_duration,
                    "success_rate": trend.success_rate,
                    "trend_direction": trend.trend_direction,
                    "trend_strength": trend.trend_strength,
                    "bottlenecks": trend.bottlenecks,
                    "optimization_suggestions": trend.optimization_suggestions,
                }
                for trend in metrics_data.trends
            ]

        return json_data

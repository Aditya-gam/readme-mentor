#!/usr/bin/env python3
"""Test script for verbosity levels."""

from app.logging import setup_logging


def test_verbosity_levels():
    """Test all verbosity levels."""
    levels = ["quiet", "normal", "verbose", "debug"]

    for level in levels:
        user_output, _ = setup_logging(user_output_level=level)
        print(f"{level.upper()}:")
        print(f"  show_progress_bars: {user_output.config.show_progress_bars}")
        print(
            f"  show_performance_metrics: {user_output.config.show_performance_metrics}"
        )
        print(f"  show_operation_steps: {user_output.config.show_operation_steps}")
        print(f"  show_raw_data: {user_output.config.show_raw_data}")
        print(f"  show_internal_state: {user_output.config.show_internal_state}")
        print()


if __name__ == "__main__":
    test_verbosity_levels()

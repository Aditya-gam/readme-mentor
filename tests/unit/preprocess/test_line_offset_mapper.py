"""Unit tests for line offset mapping functionality."""

import pytest

from app.preprocess.markdown_cleaner import LineOffsetMapper


class TestLineOffsetMapper:
    """Test line offset mapping functionality."""

    def test_single_line_text(self):
        """Test line offset mapping with single line text."""
        text = "Hello, world!"
        mapper = LineOffsetMapper(text)

        # Test line number retrieval
        assert mapper.get_line_number(0) == 0  # First character
        assert mapper.get_line_number(5) == 0  # Middle character
        assert mapper.get_line_number(12) == 0  # Last character

        # Test line range
        start_line, end_line = mapper.get_line_range(0, 12)
        assert start_line == 0
        assert end_line == 0

    def test_multi_line_text(self):
        """Test line offset mapping with multi-line text."""
        text = "Line 1\nLine 2\nLine 3"
        mapper = LineOffsetMapper(text)

        # Test line number retrieval
        assert mapper.get_line_number(0) == 0  # First line
        assert mapper.get_line_number(6) == 0  # Newline at end of first line
        assert mapper.get_line_number(7) == 1  # Second line
        assert mapper.get_line_number(13) == 1  # Newline at end of second line
        assert mapper.get_line_number(14) == 2  # Third line

        # Test line range
        start_line, end_line = mapper.get_line_range(0, 15)
        assert start_line == 0
        assert end_line == 2

    def test_line_offsets_computation(self):
        """Test line offsets computation."""
        text = "Line 1\nLine 2\nLine 3"
        mapper = LineOffsetMapper(text)

        # Verify line offsets
        assert mapper.line_offsets == [0, 7, 14]

    def test_edge_cases(self):
        """Test edge cases for line offset mapping."""
        # Empty text
        mapper = LineOffsetMapper("")
        assert mapper.get_line_number(0) == 0
        # For empty text, we can't get a line range with start_char == end_char
        with pytest.raises(ValueError, match="start_char must be less than end_char"):
            mapper.get_line_range(0, 0)

        # Text with only newlines
        mapper = LineOffsetMapper("\n\n")
        assert mapper.get_line_number(0) == 0
        assert mapper.get_line_number(1) == 1
        assert mapper.get_line_number(2) == 2

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        text = "Hello, world!"
        mapper = LineOffsetMapper(text)

        # Test negative offset
        with pytest.raises(ValueError):
            mapper.get_line_number(-1)

        # Test offset beyond text length
        with pytest.raises(ValueError):
            mapper.get_line_number(20)

        # Test invalid line range
        with pytest.raises(ValueError):
            mapper.get_line_range(5, 3)  # start > end

        with pytest.raises(ValueError):
            mapper.get_line_range(-1, 5)  # negative start

        with pytest.raises(ValueError):
            mapper.get_line_range(0, 20)  # end beyond text length

    def test_get_line_number_edge_cases(self):
        """Test get_line_number with edge cases."""
        text = "Line 1\nLine 2\nLine 3"
        mapper = LineOffsetMapper(text)

        # Test at line boundaries
        assert mapper.get_line_number(0) == 0  # Start of first line
        assert mapper.get_line_number(7) == 1  # Start of second line
        assert mapper.get_line_number(14) == 2  # Start of third line

        # Test at end of lines
        assert mapper.get_line_number(5) == 0  # End of first line
        assert mapper.get_line_number(12) == 1  # End of second line
        assert mapper.get_line_number(18) == 2  # End of third line

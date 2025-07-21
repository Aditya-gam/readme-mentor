"""
Unit tests for the citations module.

Tests the render_citations function with various scenarios including:
- Simple placeholder replacement
- Repeated placeholders for the same document
- Limiting citations to 3 unique documents
- Answer truncation to 120 words
- Edge cases and error handling
"""

import pytest
from langchain.schema import Document

from app.rag.citations import render_citations


class TestRenderCitations:
    """Test cases for the render_citations function."""

    def test_simple_placeholder_replacement(self):
        """Test that a simple placeholder gets replaced with the correct citation."""
        answer_text = "This project uses <doc_0> for dependency management."
        source_docs = [
            Document(
                page_content="Poetry is a tool for dependency management.",
                metadata={"file": "pyproject.toml", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "This project uses [pyproject.toml L1–5] for dependency management."

        assert result == expected

    def test_repeated_placeholders_same_document(self):
        """Test that repeated placeholders for the same document yield multiple citations."""
        answer_text = (
            "The project uses <doc_0> for dependencies and <doc_0> for configuration."
        )
        source_docs = [
            Document(
                page_content="Poetry configuration",
                metadata={"file": "pyproject.toml", "start_line": 1, "end_line": 10},
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "The project uses [pyproject.toml L1–10] for dependencies and [pyproject.toml L1–10] for configuration."

        assert result == expected

    def test_multiple_documents_limited_to_three(self):
        """Test that more than 3 unique document references results in only 3 citations."""
        answer_text = "This project uses <doc_0> for dependencies, <doc_1> for tests, <doc_2> for linting, and <doc_3> for docs."
        source_docs = [
            Document(
                page_content="Poetry config",
                metadata={"file": "pyproject.toml", "start_line": 1, "end_line": 5},
            ),
            Document(
                page_content="Test config",
                metadata={"file": "pytest.ini", "start_line": 1, "end_line": 3},
            ),
            Document(
                page_content="Ruff config",
                metadata={"file": "ruff.toml", "start_line": 1, "end_line": 4},
            ),
            Document(
                page_content="Documentation",
                metadata={"file": "README.md", "start_line": 1, "end_line": 10},
            ),
        ]

        result = render_citations(answer_text, source_docs)

        # Should contain citations for doc_0, doc_1, doc_2 but not doc_3
        assert "[pyproject.toml L1–5]" in result
        assert "[pytest.ini L1–3]" in result
        assert "[ruff.toml L1–4]" in result
        assert "[README.md L1–10]" not in result
        assert "<doc_3>" not in result

    def test_answer_truncation_to_120_words(self):
        """Test that answers longer than 120 words are truncated properly."""
        # Create a long answer with exactly 125 words
        long_answer = (
            "This is a very long answer that contains many words. " * 25
        )  # 125 words
        long_answer += "<doc_0> is used for configuration."

        source_docs = [
            Document(
                page_content="Configuration",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(long_answer, source_docs)

        # Should be truncated to 120 words + "..."
        words = result.split()
        # 120 words (including "..." as the last word)
        assert len(words) == 120
        assert result.endswith("...")

    def test_whitespace_handling(self):
        """Test that trailing whitespace and newlines are properly handled."""
        answer_text = "  This project uses <doc_0> for configuration.  \n\n"
        source_docs = [
            Document(
                page_content="Configuration",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "This project uses [config.py L1–5] for configuration."

        assert result == expected
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_closing_tags_removal(self):
        """Test that closing tags are properly removed."""
        answer_text = "This project uses <doc_0>configuration</doc_0> for setup."
        source_docs = [
            Document(
                page_content="Configuration",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "This project uses [config.py L1–5]configuration for setup."

        assert result == expected
        assert "</doc_0>" not in result

    def test_malformed_placeholders_handling(self):
        """Test that malformed placeholders are cleaned up."""
        answer_text = (
            "This project uses <doc_0> for config and <doc_1invalid> for tests."
        )
        source_docs = [
            Document(
                page_content="Configuration",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "This project uses [config.py L1–5] for config and  for tests."

        assert result == expected

    def test_invalid_metadata_handling(self):
        """Test that documents with invalid metadata are skipped."""
        answer_text = "This project uses <doc_0> and <doc_1> for configuration."
        source_docs = [
            Document(
                page_content="Valid config",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            ),
            Document(
                page_content="Invalid config",
                metadata={
                    "file": "invalid.py",
                    "start_line": "not_a_number",
                    "end_line": 5,
                },
            ),
        ]

        result = render_citations(answer_text, source_docs)

        # Should only contain citation for doc_0
        assert "[config.py L1–5]" in result
        assert "[invalid.py" not in result
        assert "<doc_1>" not in result

    def test_empty_answer_text(self):
        """Test that empty answer text returns empty string."""
        source_docs = [
            Document(
                page_content="Config",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations("", source_docs)
        assert result == ""

    def test_none_answer_text(self):
        """Test that None answer text returns empty string."""
        source_docs = [
            Document(
                page_content="Config",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(None, source_docs)
        assert result == ""

    def test_invalid_source_docs_type(self):
        """Test that invalid source_docs type raises TypeError."""
        answer_text = "This project uses <doc_0> for configuration."

        with pytest.raises(TypeError, match="source_docs must be a list"):
            render_citations(answer_text, "not_a_list")

    def test_document_index_out_of_bounds(self):
        """Test that document indices beyond source_docs length are ignored."""
        answer_text = "This project uses <doc_0> and <doc_5> for configuration."
        source_docs = [
            Document(
                page_content="Config",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            )
        ]

        result = render_citations(answer_text, source_docs)

        # Should only contain citation for doc_0
        assert "[config.py L1–5]" in result
        assert "<doc_5>" not in result

    def test_mixed_valid_and_invalid_placeholders(self):
        """Test handling of mixed valid and invalid placeholders."""
        answer_text = "This project uses <doc_0> for config, <doc_1> for tests, and <doc_invalid> for docs."
        source_docs = [
            Document(
                page_content="Config",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            ),
            Document(
                page_content="Tests",
                metadata={"file": "test.py", "start_line": 1, "end_line": 10},
            ),
        ]

        result = render_citations(answer_text, source_docs)

        # Should contain citations for doc_0 and doc_1, but not doc_invalid
        assert "[config.py L1–5]" in result
        assert "[test.py L1–10]" in result
        assert "<doc_invalid>" not in result

    def test_string_line_numbers_conversion(self):
        """Test that string line numbers are properly converted to integers."""
        answer_text = "This project uses <doc_0> for configuration."
        source_docs = [
            Document(
                page_content="Config",
                metadata={"file": "config.py", "start_line": "1", "end_line": "5"},
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "This project uses [config.py L1–5] for configuration."

        assert result == expected

    def test_missing_metadata_fields(self):
        """Test that documents with missing metadata fields are skipped."""
        answer_text = "This project uses <doc_0> and <doc_1> for configuration."
        source_docs = [
            Document(
                page_content="Valid config",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            ),
            Document(
                page_content="Invalid config",
                # Missing start_line and end_line
                metadata={"file": "invalid.py"},
            ),
        ]

        result = render_citations(answer_text, source_docs)

        # Should only contain citation for doc_0
        assert "[config.py L1–5]" in result
        assert "[invalid.py" not in result
        assert "<doc_1>" not in result

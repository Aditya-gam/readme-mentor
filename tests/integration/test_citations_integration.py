"""
Integration tests for the citations module with the RAG chain.

Tests that the citations module integrates properly with the document
processing pipeline and RAG chain functionality.
"""

from langchain.schema import Document

from app.rag.citations import render_citations


class TestCitationsIntegration:
    """Integration tests for citations with RAG chain components."""

    def test_citations_with_real_document_metadata(self):
        """Test citations with realistic document metadata from the ingestion pipeline."""
        answer_text = (
            "This project uses <doc_0> for configuration and <doc_1> for testing."
        )

        # Simulate documents with metadata from the ingestion pipeline
        source_docs = [
            Document(
                page_content="[tool.poetry]\nname = 'readme-mentor'\nversion = '0.2.0'",
                metadata={
                    "file": "pyproject.toml",
                    "start_line": 1,
                    "end_line": 3,
                    "chunk_id": 0,
                    "source": "pyproject.toml",
                },
            ),
            Document(
                page_content="[tool.pytest.ini_options]\ntestpaths = ['tests']\npython_files = ['test_*.py']",
                metadata={
                    "file": "pyproject.toml",
                    "start_line": 50,
                    "end_line": 52,
                    "chunk_id": 1,
                    "source": "pyproject.toml",
                },
            ),
        ]

        result = render_citations(answer_text, source_docs)
        expected = "This project uses [pyproject.toml L1–3] for configuration and [pyproject.toml L50–52] for testing."

        assert result == expected

    def test_citations_with_mixed_document_sources(self):
        """Test citations with documents from different source files."""
        answer_text = "The project uses <doc_0> for dependencies, <doc_1> for linting, and <doc_2> for documentation."

        source_docs = [
            Document(
                page_content="dependencies = ['fastapi', 'uvicorn']",
                metadata={"file": "pyproject.toml", "start_line": 10, "end_line": 12},
            ),
            Document(
                page_content="[tool.ruff]\nline-length = 88",
                metadata={"file": "ruff.toml", "start_line": 1, "end_line": 2},
            ),
            Document(
                page_content="# README Mentor\n\nAI-powered README generation tool.",
                metadata={"file": "README.md", "start_line": 1, "end_line": 3},
            ),
        ]

        result = render_citations(answer_text, source_docs)

        # Should contain all three citations
        assert "[pyproject.toml L10–12]" in result
        assert "[ruff.toml L1–2]" in result
        assert "[README.md L1–3]" in result

    def test_citations_with_chain_formatting(self):
        """Test that citations work with the document formatting used by the RAG chain."""
        # Simulate an LLM answer that references documents
        answer_text = (
            "Based on the documentation, <doc_0> and the code follows <doc_1>."
        )

        source_docs = [
            Document(
                page_content="This project uses Poetry for dependency management.",
                metadata={"file": "pyproject.toml", "start_line": 1, "end_line": 5},
            ),
            Document(
                page_content="The project follows PEP 8 style guidelines.",
                metadata={"file": "ruff.toml", "start_line": 1, "end_line": 3},
            ),
        ]

        result = render_citations(answer_text, source_docs)
        expected = "Based on the documentation, [pyproject.toml L1–5] and the code follows [ruff.toml L1–3]."

        assert result == expected

    def test_citations_with_edge_case_metadata(self):
        """Test citations with edge cases in document metadata."""
        answer_text = "The project uses <doc_0> for configuration."

        # Test with string line numbers (should be converted to int)
        source_docs = [
            Document(
                page_content="Configuration content",
                metadata={
                    "file": "config.py",
                    "start_line": "1",
                    "end_line": "10",
                    "chunk_id": "0",
                },
            )
        ]

        result = render_citations(answer_text, source_docs)
        expected = "The project uses [config.py L1–10] for configuration."

        assert result == expected

    def test_citations_with_large_answer_truncation(self):
        """Test that long answers are properly truncated."""
        # Create a long answer that would exceed 120 words
        long_answer = (
            "This is a very detailed explanation of the project structure. " * 30
        )  # ~150 words
        long_answer += (
            "The project uses <doc_0> for configuration and <doc_1> for testing."
        )

        source_docs = [
            Document(
                page_content="Configuration",
                metadata={"file": "config.py", "start_line": 1, "end_line": 5},
            ),
            Document(
                page_content="Testing",
                metadata={"file": "test.py", "start_line": 1, "end_line": 10},
            ),
        ]

        result = render_citations(long_answer, source_docs)

        # Should be truncated to 120 words
        words = result.split()
        assert len(words) <= 120
        assert result.endswith("...")
        # Citations at the end may be truncated, which is correct behavior

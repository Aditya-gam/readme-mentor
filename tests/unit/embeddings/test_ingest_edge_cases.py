"""Edge case tests for ingest functionality."""

from unittest.mock import Mock, patch

from app.embeddings.ingest import (
    _chunk_text_with_line_mapping,
    _extract_repo_slug,
    _generate_collection_name,
    _generate_embeddings_batch,
    _process_file_for_chunking,
)


class TestExtractRepoSlugEdgeCases:
    """Test edge cases for repository slug extraction."""

    def test_extract_repo_slug_standard_url(self):
        """Test extraction from standard GitHub URL."""
        result = _extract_repo_slug("https://github.com/octocat/Hello-World")
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_with_git_suffix(self):
        """Test extraction from URL with .git suffix."""
        result = _extract_repo_slug("https://github.com/octocat/Hello-World.git")
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_with_trailing_slash(self):
        """Test extraction from URL with trailing slash."""
        result = _extract_repo_slug("https://github.com/octocat/Hello-World/")
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_with_extra_paths(self):
        """Test extraction from URL with extra paths."""
        result = _extract_repo_slug("https://github.com/octocat/Hello-World/tree/main")
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_with_query_params(self):
        """Test extraction from URL with query parameters."""
        result = _extract_repo_slug(
            "https://github.com/octocat/Hello-World?param=value"
        )
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_with_fragment(self):
        """Test extraction from URL with fragment."""
        result = _extract_repo_slug("https://github.com/octocat/Hello-World#section")
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_http_protocol(self):
        """Test extraction from HTTP URL."""
        result = _extract_repo_slug("http://github.com/octocat/Hello-World")
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_uppercase_domain(self):
        """Test extraction from URL with uppercase domain."""
        result = _extract_repo_slug("https://GITHUB.COM/octocat/Hello-World")
        assert result == "GITHUB.COM_octocat_Hello-World"

    def test_extract_repo_slug_short_url(self):
        """Test extraction from short URL."""
        result = _extract_repo_slug("https://github.com/octocat")
        assert result == "github.com_octocat"

    def test_extract_repo_slug_non_github_url(self):
        """Test extraction from non-GitHub URL."""
        result = _extract_repo_slug("https://gitlab.com/user/repo")
        assert result == "gitlab.com_user_repo"

    def test_extract_repo_slug_invalid_url(self):
        """Test extraction from invalid URL."""
        result = _extract_repo_slug("not-a-url")
        assert result == "not-a-url"


class TestCreatePersistDirectoryEdgeCases:
    """Test edge cases for persist directory creation."""

    pass


class TestGenerateCollectionNameEdgeCases:
    """Test edge cases for collection name generation."""

    def test_generate_collection_name_success(self, mock_uuid):
        """Test successful collection name generation."""
        # Call function
        result = _generate_collection_name("test_repo")

        # Verify result
        assert result == "test_repo_12345678"

    def test_generate_collection_name_with_special_chars(self, mock_uuid):
        """Test collection name generation with special characters."""
        # Call function
        result = _generate_collection_name("user-name_repo_name")

        # Verify result
        assert result == "user-name_repo_name_12345678"


class TestChunkTextWithLineMappingEdgeCases:
    """Test edge cases for text chunking with line mapping."""

    def test_chunk_text_with_line_mapping_empty_text(self):
        """Test chunking empty text."""
        from pathlib import Path

        result = _chunk_text_with_line_mapping("", Path("test.md"))

        # Should return empty list for empty text
        assert result == []

    def test_chunk_text_with_line_mapping_single_character(self):
        """Test chunking single character."""
        from pathlib import Path

        result = _chunk_text_with_line_mapping("a", Path("test.md"))

        # Should return one document
        assert len(result) == 1
        assert result[0].page_content == "a"
        assert result[0].metadata["file"] == "test.md"
        assert result[0].metadata["start_line"] == 0
        assert result[0].metadata["end_line"] == 0

    def test_chunk_text_with_line_mapping_single_line(self):
        """Test chunking single line."""
        from pathlib import Path

        text = "This is a single line of text."
        result = _chunk_text_with_line_mapping(text, Path("test.md"))

        # Should return one document
        assert len(result) == 1
        assert result[0].page_content == text
        assert result[0].metadata["file"] == "test.md"
        assert result[0].metadata["start_line"] == 0
        assert result[0].metadata["end_line"] == 0

    def test_chunk_text_with_line_mapping_multiple_lines(self):
        """Test chunking multiple lines."""
        from pathlib import Path

        text = "Line 1\nLine 2\nLine 3"
        result = _chunk_text_with_line_mapping(text, Path("test.md"))

        # Should return one document for short text
        assert len(result) == 1
        assert result[0].page_content == text
        assert result[0].metadata["file"] == "test.md"
        assert result[0].metadata["start_line"] == 0
        assert result[0].metadata["end_line"] == 2

    def test_chunk_text_with_line_mapping_large_text(self):
        """Test chunking large text."""
        from pathlib import Path

        # Create text larger than chunk size
        text = "This is a very long line of text. " * 100
        result = _chunk_text_with_line_mapping(
            text, Path("test.md"), chunk_size=200, chunk_overlap=50
        )

        # Should return multiple documents
        assert len(result) > 1
        for doc in result:
            assert doc.metadata["file"] == "test.md"
            assert "start_line" in doc.metadata
            assert "end_line" in doc.metadata
            assert "chunk_start" in doc.metadata
            assert "chunk_end" in doc.metadata

    def test_chunk_text_with_line_mapping_with_overlap(self):
        """Test chunking with overlap."""
        from pathlib import Path

        text = "This is a test document with multiple lines for testing chunking with overlap."
        result = _chunk_text_with_line_mapping(
            text, Path("test.md"), chunk_size=50, chunk_overlap=20
        )

        # Should return multiple documents due to overlap
        assert len(result) > 1
        for doc in result:
            assert doc.metadata["file"] == "test.md"
            assert "start_line" in doc.metadata
            assert "end_line" in doc.metadata


class TestProcessFileForChunkingEdgeCases:
    """Test edge cases for file processing for chunking."""

    def test_process_file_for_chunking_exception(self):
        """Test file processing with exception."""
        from pathlib import Path

        # Setup mock to raise exception
        mock_clean = Mock(side_effect=Exception("File not found"))

        with patch("app.embeddings.ingest.clean_markdown_file", mock_clean):
            # Call function
            result = _process_file_for_chunking(Path("nonexistent.md"))

        # Verify result is empty list on exception
        assert result == []


class TestGenerateEmbeddingsBatchEdgeCases:
    """Test edge cases for embedding batch generation."""

    def test_generate_embeddings_batch_empty_documents(self):
        """Test embedding generation with empty documents."""
        mock_model = Mock()

        result = _generate_embeddings_batch([], mock_model)

        # Should return empty list
        assert result == []
        mock_model.embed_documents.assert_not_called()

    def test_generate_embeddings_batch_single_document(self):
        """Test embedding generation with single document."""
        from langchain_core.documents import Document

        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        documents = [Document(page_content="single document")]

        result = _generate_embeddings_batch(documents, mock_model)

        # Verify result
        assert result == [[0.1, 0.2, 0.3]]
        mock_model.embed_documents.assert_called_once_with(["single document"])

    def test_generate_embeddings_batch_large_batch_size(self):
        """Test embedding generation with large batch size."""
        from langchain_core.documents import Document

        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        documents = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]

        result = _generate_embeddings_batch(documents, mock_model, batch_size=100)

        # Verify result
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.embed_documents.assert_called_once_with(["doc1", "doc2"])

    def test_generate_embeddings_batch_model_exception(self):
        """Test embedding generation with model exception."""
        from langchain_core.documents import Document

        mock_model = Mock()
        mock_model.embed_documents.side_effect = Exception("Model error")

        documents = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]

        result = _generate_embeddings_batch(documents, mock_model)

        # Should return zero vectors as fallback
        assert len(result) == 2
        # Default MiniLM dimension
        assert all(len(embedding) == 384 for embedding in result)
        assert all(all(val == 0.0 for val in embedding) for embedding in result)

    def test_generate_embeddings_batch_partial_failure(self):
        """Test embedding generation with partial failure."""
        from langchain_core.documents import Document

        mock_model = Mock()
        # First call succeeds, second call fails
        mock_model.embed_documents.side_effect = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            Exception("Model error"),
        ]

        documents = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
            Document(page_content="doc4"),
        ]

        result = _generate_embeddings_batch(documents, mock_model, batch_size=2)

        # Should return embeddings for first batch and zeros for second batch
        assert len(result) == 4
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert all(len(embedding) == 384 for embedding in result[2:])
        assert all(all(val == 0.0 for val in embedding) for embedding in result[2:])


class TestIngestRepositoryEdgeCases:
    """Test edge cases for repository ingestion."""

    pass

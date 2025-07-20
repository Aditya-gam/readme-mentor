"""Unit tests for the ingestion pipeline.

This module tests individual functions in the ingestion pipeline with proper mocking
to achieve high test coverage without external dependencies.
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.embeddings.ingest import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    _chunk_text_with_line_mapping,
    _create_persist_directory,
    _extract_repo_slug,
    _generate_collection_name,
    _generate_embeddings_batch,
    _process_file_for_chunking,
    ingest_repository,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestExtractRepoSlug:
    """Test the _extract_repo_slug function."""

    def test_extract_repo_slug_github_url(self):
        """Test extracting repo slug from GitHub URL."""
        url = "https://github.com/octocat/Hello-World"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_github_url_with_git(self):
        """Test extracting repo slug from GitHub URL with .git suffix."""
        url = "https://github.com/octocat/Hello-World.git"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_github_url_with_trailing_slash(self):
        """Test extracting repo slug from GitHub URL with trailing slash."""
        url = "https://github.com/octocat/Hello-World/"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_non_github_url(self):
        """Test extracting repo slug from non-GitHub URL."""
        url = "https://gitlab.com/user/repo"
        result = _extract_repo_slug(url)
        assert result == "gitlab.com_user_repo"

    def test_extract_repo_slug_http_url(self):
        """Test extracting repo slug from HTTP URL."""
        url = "http://github.com/octocat/Hello-World"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_short_url(self):
        """Test extracting repo slug from short URL."""
        url = "https://github.com/octocat"
        result = _extract_repo_slug(url)
        assert result == "github.com_octocat"


class TestCreatePersistDirectory:
    """Test the _create_persist_directory function."""

    @patch("app.embeddings.ingest.Path")
    def test_create_persist_directory(self, mock_path):
        """Test creating persist directory."""
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)

        result = _create_persist_directory("test_repo")

        mock_path.assert_called_once_with("data")
        mock_path_instance.__truediv__.assert_called()
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result == mock_path_instance


class TestGenerateCollectionName:
    """Test the _generate_collection_name function."""

    @patch("app.embeddings.ingest.uuid")
    def test_generate_collection_name(self, mock_uuid):
        """Test generating collection name."""
        mock_uuid.uuid4.return_value = "12345678-1234-1234-1234-123456789abc"

        result = _generate_collection_name("test_repo")

        assert result == "test_repo_12345678"


class TestChunkTextWithLineMapping:
    """Test the _chunk_text_with_line_mapping function."""

    @patch("app.embeddings.ingest.LineOffsetMapper")
    @patch("app.embeddings.ingest.RecursiveCharacterTextSplitter")
    def test_chunk_text_with_line_mapping(self, mock_splitter_class, mock_mapper_class):
        """Test chunking text with line mapping."""
        # Setup mocks
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.get_line_range.return_value = (1, 5)

        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]

        # Test data
        text = "This is a test document\nwith multiple lines\nfor chunking"
        file_path = Path("test.md")

        # Call function
        result = _chunk_text_with_line_mapping(
            text, file_path, chunk_size=10, chunk_overlap=2
        )

        # Verify mocks were called correctly
        mock_mapper_class.assert_called_once_with(text)
        mock_splitter_class.assert_called_once_with(
            chunk_size=10,
            chunk_overlap=2,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        mock_splitter.split_text.assert_called_once_with(text)

        # Verify results
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)

        # Check first document
        doc1 = result[0]
        assert doc1.page_content == "chunk1"
        assert doc1.metadata["file"] == str(file_path)
        assert doc1.metadata["start_line"] == 1
        assert doc1.metadata["end_line"] == 5
        assert "chunk_start" in doc1.metadata
        assert "chunk_end" in doc1.metadata

    @patch("app.embeddings.ingest.LineOffsetMapper")
    @patch("app.embeddings.ingest.RecursiveCharacterTextSplitter")
    def test_chunk_text_with_defaults(self, mock_splitter_class, mock_mapper_class):
        """Test chunking text with default parameters."""
        # Setup mocks
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.get_line_range.return_value = (1, 3)

        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["single chunk"]

        # Test data
        text = "Simple text"
        file_path = Path("simple.md")

        # Call function with defaults
        result = _chunk_text_with_line_mapping(text, file_path)

        # Verify default parameters were used
        mock_splitter_class.assert_called_once_with(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        assert len(result) == 1
        assert result[0].page_content == "single chunk"


class TestProcessFileForChunking:
    """Test the _process_file_for_chunking function."""

    @patch("app.embeddings.ingest.clean_markdown_file")
    @patch("app.embeddings.ingest._chunk_text_with_line_mapping")
    def test_process_file_success(self, mock_chunk, mock_clean):
        """Test successful file processing."""
        # Setup mocks
        mock_cleaned_doc = Mock()
        mock_cleaned_doc.text = "cleaned content"
        mock_clean.return_value = mock_cleaned_doc

        mock_documents = [
            Document(page_content="chunk1"),
            Document(page_content="chunk2"),
        ]
        mock_chunk.return_value = mock_documents

        # Test data
        file_path = Path("test.md")

        # Call function
        result = _process_file_for_chunking(file_path)

        # Verify mocks were called
        mock_clean.assert_called_once_with(file_path, include_code=False)
        mock_chunk.assert_called_once_with("cleaned content", file_path)

        # Verify result
        assert result == mock_documents

    @patch("app.embeddings.ingest.clean_markdown_file")
    def test_process_file_exception(self, mock_clean):
        """Test file processing with exception."""
        # Setup mock to raise exception
        mock_clean.side_effect = Exception("File not found")

        # Test data
        file_path = Path("nonexistent.md")

        # Call function
        result = _process_file_for_chunking(file_path)

        # Verify result is empty list on exception
        assert result == []


class TestGenerateEmbeddingsBatch:
    """Test the _generate_embeddings_batch function."""

    def test_generate_embeddings_batch_success(self):
        """Test successful batch embedding generation."""
        # Setup mock embedding model
        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Test documents
        documents = [Document(page_content="doc1"), Document(page_content="doc2")]

        # Call function
        result = _generate_embeddings_batch(documents, mock_model, batch_size=2)

        # Verify result
        assert len(result) == 2
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Verify model was called correctly
        mock_model.embed_documents.assert_called_once_with(["doc1", "doc2"])

    def test_generate_embeddings_batch_multiple_batches(self):
        """Test embedding generation with multiple batches."""
        # Setup mock embedding model
        mock_model = Mock()
        mock_model.embed_documents.side_effect = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # First batch
            [[0.7, 0.8, 0.9]],  # Second batch
        ]

        # Test documents
        documents = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]

        # Call function
        result = _generate_embeddings_batch(documents, mock_model, batch_size=2)

        # Verify result
        assert len(result) == 3
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        # Verify model was called twice
        assert mock_model.embed_documents.call_count == 2
        mock_model.embed_documents.assert_any_call(["doc1", "doc2"])
        mock_model.embed_documents.assert_any_call(["doc3"])

    def test_generate_embeddings_batch_exception_fallback(self):
        """Test embedding generation with exception fallback."""
        # Setup mock embedding model to raise exception
        mock_model = Mock()
        mock_model.embed_documents.side_effect = Exception("Model error")

        # Test documents
        documents = [Document(page_content="doc1"), Document(page_content="doc2")]

        # Call function
        result = _generate_embeddings_batch(documents, mock_model, batch_size=2)

        # Verify fallback zero vectors
        assert len(result) == 2
        # Default MiniLM dimension
        assert all(len(embedding) == 384 for embedding in result)
        assert all(all(val == 0.0 for val in embedding) for embedding in result)


class TestIngestRepository:
    """Test the ingest_repository function."""

    @patch("app.embeddings.ingest._generate_collection_name")
    @patch("app.embeddings.ingest.Chroma")
    @patch("app.embeddings.ingest.HuggingFaceEmbeddings")
    @patch("app.embeddings.ingest._generate_embeddings_batch")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_success(
        self,
        mock_path,
        mock_extract_slug,
        mock_fetch_files,
        mock_process_file,
        mock_generate_embeddings,
        mock_embeddings_class,
        mock_chroma_class,
        mock_generate_collection,
    ):
        """Test successful repository ingestion."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md", "file2.md"]
        mock_process_file.side_effect = [
            [Document(page_content="doc1"), Document(page_content="doc2")],
            [Document(page_content="doc3")],
        ]
        mock_generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_generate_collection.return_value = "test_collection"

        # Mock Path.exists() to return True for all files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function
        result = ingest_repository("https://github.com/test/repo")

        # Verify mocks were called correctly
        mock_extract_slug.assert_called_once_with("https://github.com/test/repo")
        mock_fetch_files.assert_called_once_with(
            "https://github.com/test/repo", ("README*", "docs/**/*.md")
        )
        assert mock_process_file.call_count == 2

        # Verify embedding model initialization
        mock_embeddings_class.assert_called_once_with(
            model_name=DEFAULT_EMBEDDING_MODEL
        )

        # Verify embeddings generation
        all_docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        mock_generate_embeddings.assert_called_once_with(
            all_docs, mock_embedding_model, DEFAULT_BATCH_SIZE
        )

        # Verify ChromaDB creation
        mock_chroma_class.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embedding_model,
        )

        # Verify documents were added to vector store
        mock_vectorstore.add_texts.assert_called_once()
        call_args = mock_vectorstore.add_texts.call_args
        assert call_args[1]["texts"] == ["doc1", "doc2", "doc3"]
        assert len(call_args[1]["metadatas"]) == 3

        # Verify result
        assert result == mock_vectorstore

    @patch("app.embeddings.ingest._extract_repo_slug")
    def test_ingest_repository_no_files_found(self, mock_extract_slug):
        """Test repository ingestion with no files found."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"

        with patch("app.embeddings.ingest.fetch_repository_files") as mock_fetch_files:
            mock_fetch_files.return_value = []

            # Call function and expect exception
            from app.embeddings.ingest import ingest_repository

            with pytest.raises(Exception, match="No files found in repository"):
                ingest_repository("https://github.com/test/repo")

    @patch("app.embeddings.ingest._extract_repo_slug")
    def test_ingest_repository_no_documents_created(self, mock_extract_slug):
        """Test repository ingestion with no documents created."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"

        with (
            patch("app.embeddings.ingest.fetch_repository_files") as mock_fetch_files,
            patch(
                "app.embeddings.ingest._process_file_for_chunking"
            ) as mock_process_file,
            patch("app.embeddings.ingest.Path") as mock_path,
        ):
            mock_fetch_files.return_value = ["file1.md"]
            mock_process_file.return_value = []

            # Mock Path.exists() to return True
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Call function and expect exception
            from app.embeddings.ingest import ingest_repository

            with pytest.raises(Exception, match="No documents created from files"):
                ingest_repository("https://github.com/test/repo")

    @patch("app.embeddings.ingest._generate_collection_name")
    @patch("app.embeddings.ingest.Chroma")
    @patch("app.embeddings.ingest.HuggingFaceEmbeddings")
    @patch("app.embeddings.ingest._generate_embeddings_batch")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_embedding_mismatch(
        self,
        mock_path,
        mock_extract_slug,
        mock_fetch_files,
        mock_process_file,
        mock_generate_embeddings,
        mock_embeddings_class,
        mock_chroma_class,
        mock_generate_collection,
    ):
        """Test repository ingestion with embedding count mismatch."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md"]
        mock_process_file.return_value = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]
        mock_generate_embeddings.return_value = [
            [0.1, 0.2, 0.3]
        ]  # Only one embedding for two docs
        mock_generate_collection.return_value = "test_collection"

        # Mock Path.exists() to return True for all files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function and expect exception
        with pytest.raises(Exception, match="Embedding generation failed"):
            ingest_repository("https://github.com/test/repo")

    @patch("app.embeddings.ingest._generate_collection_name")
    @patch("app.embeddings.ingest.Chroma")
    @patch("app.embeddings.ingest.HuggingFaceEmbeddings")
    @patch("app.embeddings.ingest._generate_embeddings_batch")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_with_custom_parameters(
        self,
        mock_path,
        mock_extract_slug,
        mock_fetch_files,
        mock_process_file,
        mock_generate_embeddings,
        mock_embeddings_class,
        mock_chroma_class,
        mock_generate_collection,
    ):
        """Test repository ingestion with custom parameters."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md"]
        mock_process_file.return_value = [Document(page_content="doc1")]
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_generate_collection.return_value = "test_collection"

        # Mock Path.exists() to return True for all files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function with custom parameters
        result = ingest_repository(
            "https://github.com/test/repo",
            file_glob=("*.md",),
            chunk_size=512,
            chunk_overlap=64,
            batch_size=32,
            embedding_model_name="custom-model",
            collection_name="custom_collection",
            persist_directory="/tmp/test",
        )

        # Verify custom parameters were used
        mock_fetch_files.assert_called_once_with(
            "https://github.com/test/repo", ("*.md",)
        )
        mock_embeddings_class.assert_called_once_with(model_name="custom-model")
        mock_generate_embeddings.assert_called_once_with(
            [Document(page_content="doc1")], mock_embedding_model, 32
        )

        # Verify ChromaDB was created with custom collection name and persist directory
        mock_chroma_class.assert_called_once_with(
            collection_name="custom_collection",
            embedding_function=mock_embedding_model,
            persist_directory="/tmp/test",
        )

        assert result == mock_vectorstore

    @patch("app.embeddings.ingest._generate_collection_name")
    @patch("app.embeddings.ingest.Chroma")
    @patch("app.embeddings.ingest.HuggingFaceEmbeddings")
    @patch("app.embeddings.ingest._generate_embeddings_batch")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_with_file_glob_none(
        self,
        mock_path,
        mock_extract_slug,
        mock_fetch_files,
        mock_process_file,
        mock_generate_embeddings,
        mock_embeddings_class,
        mock_chroma_class,
        mock_generate_collection,
    ):
        """Test repository ingestion with file_glob=None (default patterns)."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md"]
        mock_process_file.return_value = [Document(page_content="doc1")]

        # Mock Path.exists() to return True for all files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_generate_collection.return_value = "test_collection"

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function with file_glob=None
        result = ingest_repository("https://github.com/test/repo", file_glob=None)

        # Verify default file patterns were used
        mock_fetch_files.assert_called_once_with(
            "https://github.com/test/repo", ("README*", "docs/**/*.md")
        )

        assert result == mock_vectorstore


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])

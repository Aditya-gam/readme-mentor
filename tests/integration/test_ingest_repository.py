"""Unit tests for the main ingest_repository function."""

import logging
from unittest.mock import Mock, patch

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Mock Path.exists() to return True for the files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_process_file.side_effect = [
            [Mock(page_content="doc1"), Mock(page_content="doc2")],
            [Mock(page_content="doc3")],
        ]
        mock_generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_generate_collection.return_value = "test_collection"

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function
        from app.embeddings.ingest import (
            DEFAULT_BATCH_SIZE,
            DEFAULT_EMBEDDING_MODEL,
            ingest_repository,
        )

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
        mock_generate_embeddings.assert_called_once()
        call_args = mock_generate_embeddings.call_args
        assert len(call_args[0][0]) == 3  # Should have 3 documents
        assert call_args[0][1] == mock_embedding_model
        assert call_args[0][2] == DEFAULT_BATCH_SIZE

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
    @patch("app.embeddings.ingest.fetch_repository_files")
    def test_ingest_repository_no_files_found(
        self, mock_fetch_files, mock_extract_slug
    ):
        """Test repository ingestion with no files found."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = []

        # Call function and expect exception
        from app.embeddings.ingest import ingest_repository

        with pytest.raises(Exception, match="No files found in repository"):
            ingest_repository("https://github.com/test/repo")

    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_no_documents_created(
        self, mock_path, mock_process_file, mock_fetch_files, mock_extract_slug
    ):
        """Test repository ingestion with no documents created."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["README.md"]

        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_process_file.return_value = []

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

        # Mock Path.exists() to return True for the files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_process_file.return_value = [
            Mock(page_content="doc1"),
            Mock(page_content="doc2"),
        ]
        mock_generate_embeddings.return_value = [
            [0.1, 0.2, 0.3]
        ]  # Only one embedding for two docs
        mock_generate_collection.return_value = "test_collection"

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function and expect exception
        from app.embeddings.ingest import ingest_repository

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

        # Mock Path.exists() to return True for the files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_process_file.return_value = [Mock(page_content="doc1")]
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_generate_collection.return_value = "test_collection"

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function with custom parameters
        from app.embeddings.ingest import ingest_repository

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
        mock_generate_embeddings.assert_called_once()
        call_args = mock_generate_embeddings.call_args
        assert len(call_args[0][0]) == 1  # Should have 1 document
        assert call_args[0][1] == mock_embedding_model
        assert call_args[0][2] == 32

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

        # Mock Path.exists() to return True for the files
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_process_file.return_value = [Mock(page_content="doc1")]
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_generate_collection.return_value = "test_collection"

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        # Call function with file_glob=None
        from app.embeddings.ingest import ingest_repository

        result = ingest_repository("https://github.com/test/repo", file_glob=None)

        # Verify default file patterns were used
        mock_fetch_files.assert_called_once_with(
            "https://github.com/test/repo", ("README*", "docs/**/*.md")
        )

        assert result == mock_vectorstore

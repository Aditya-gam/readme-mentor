"""Unit tests for the main ingest_repository function."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.embeddings.ingest import ingest_repository


class TestIngestRepository:
    """Test the ingest_repository function."""

    @patch("app.embeddings.ingest._generate_collection_name")
    @patch("app.embeddings.ingest.Chroma")
    @patch("app.embeddings.ingest.HuggingFaceEmbeddings")
    @patch("app.embeddings.ingest._generate_embeddings_batch")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.validate_repo_url")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_success(
        self,
        mock_path,
        mock_validate_url,
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
        mock_validate_url.return_value = "https://github.com/test/repo"
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md", "file2.md"]

        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        # Create documents that will be processed - one document per file
        mock_documents_file1 = [
            Document(page_content="chunk1", metadata={"file": "file1.md"})
        ]
        mock_documents_file2 = [
            Document(page_content="chunk2", metadata={"file": "file2.md"})
        ]
        mock_process_file.side_effect = [mock_documents_file1, mock_documents_file2]

        # Create embeddings that match the total document count (2 documents)
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_generate_embeddings.return_value = mock_embeddings

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        mock_generate_collection.return_value = "test_collection"

        # Call function
        result = ingest_repository("https://github.com/test/repo")

        # Verify mocks were called correctly
        mock_validate_url.assert_called_once_with("https://github.com/test/repo")
        mock_extract_slug.assert_called_once_with("https://github.com/test/repo")
        mock_fetch_files.assert_called_once()
        assert mock_process_file.call_count == 2
        mock_generate_embeddings.assert_called_once()
        mock_embeddings_class.assert_called_once()
        mock_chroma_class.assert_called_once()
        mock_vectorstore.add_texts.assert_called_once()

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
        mock_fetch_files.return_value = ["file1.md"]

        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_process_file.return_value = []

        # Call function and expect exception
        with pytest.raises(Exception, match="No documents created from files"):
            ingest_repository("https://github.com/test/repo")

    @patch("app.embeddings.ingest._generate_collection_name")
    @patch("app.embeddings.ingest.Chroma")
    @patch("app.embeddings.ingest.HuggingFaceEmbeddings")
    @patch("app.embeddings.ingest._generate_embeddings_batch")
    @patch("app.embeddings.ingest._process_file_for_chunking")
    @patch("app.embeddings.ingest.fetch_repository_files")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("app.embeddings.ingest.validate_repo_url")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_embedding_mismatch(
        self,
        mock_path,
        mock_validate_url,
        mock_extract_slug,
        mock_fetch_files,
        mock_process_file,
        mock_generate_embeddings,
        mock_embeddings_class,
        mock_chroma_class,
        mock_generate_collection,
    ):
        """Test repository ingestion with embedding mismatch."""
        # Setup mocks
        mock_validate_url.return_value = "https://github.com/test/repo"
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md"]

        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        # Create two documents
        mock_documents = [
            Document(page_content="chunk1"),
            Document(page_content="chunk2"),
        ]
        mock_process_file.return_value = mock_documents

        # Return only one embedding for two documents (mismatch)
        # Only one embedding for two documents
        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_generate_embeddings.return_value = mock_embeddings

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        mock_generate_collection.return_value = "test_collection"

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
    @patch("app.embeddings.ingest.validate_repo_url")
    @patch("app.embeddings.ingest.Path")
    def test_ingest_repository_with_custom_parameters(
        self,
        mock_path,
        mock_validate_url,
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
        mock_validate_url.return_value = "https://github.com/test/repo"
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md"]

        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_documents = [Document(page_content="chunk1")]
        mock_process_file.return_value = mock_documents

        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_generate_embeddings.return_value = mock_embeddings

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        mock_generate_collection.return_value = "test_collection"

        # Call function with custom parameters
        result = ingest_repository(
            "https://github.com/test/repo",
            chunk_size=512,
            chunk_overlap=64,
            batch_size=32,
            embedding_model_name="custom-model",
        )

        # Verify custom parameters were used
        mock_embeddings_class.assert_called_once_with(model_name="custom-model")
        mock_generate_embeddings.assert_called_once_with(
            mock_documents, mock_embedding_model, 32
        )

        # Verify result
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
        """Test repository ingestion with file_glob=None."""
        # Setup mocks
        mock_extract_slug.return_value = "test_repo"
        mock_fetch_files.return_value = ["file1.md"]

        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True

        mock_documents = [Document(page_content="chunk1")]
        mock_process_file.return_value = mock_documents

        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_generate_embeddings.return_value = mock_embeddings

        mock_embedding_model = Mock()
        mock_embeddings_class.return_value = mock_embedding_model

        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore

        mock_generate_collection.return_value = "test_collection"

        # Call function with file_glob=None
        result = ingest_repository("https://github.com/test/repo", file_glob=None)

        # Verify default file patterns were used
        mock_fetch_files.assert_called_once_with(
            "https://github.com/test/repo", ("README*", "docs/**/*.md")
        )

        # Verify result
        assert result == mock_vectorstore

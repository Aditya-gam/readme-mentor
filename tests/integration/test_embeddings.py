"""Unit tests for embedding generation functionality."""

import logging
from unittest.mock import Mock

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGenerateEmbeddingsBatch:
    """Test the _generate_embeddings_batch function."""

    def test_generate_embeddings_batch_success(self):
        """Test successful batch embedding generation."""
        # Setup mock embedding model
        mock_model = Mock()
        mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Test documents
        documents = [Mock(page_content="doc1"), Mock(page_content="doc2")]

        # Call function
        from app.embeddings.ingest import _generate_embeddings_batch

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
            Mock(page_content="doc1"),
            Mock(page_content="doc2"),
            Mock(page_content="doc3"),
        ]

        # Call function
        from app.embeddings.ingest import _generate_embeddings_batch

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
        documents = [Mock(page_content="doc1"), Mock(page_content="doc2")]

        # Call function
        from app.embeddings.ingest import _generate_embeddings_batch

        result = _generate_embeddings_batch(documents, mock_model, batch_size=2)

        # Verify fallback zero vectors
        assert len(result) == 2
        # Default MiniLM dimension
        assert all(len(embedding) == 384 for embedding in result)
        assert all(all(val == 0.0 for val in embedding) for embedding in result)

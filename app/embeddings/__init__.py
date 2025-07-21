"""Embeddings and vector store functionality for readme-mentor.

This package provides text chunking, embedding generation, and vector store
integration for processing repository content.
"""

from .ingest import get_embedding_model, get_vector_store, ingest_repository

__all__ = ["ingest_repository", "get_embedding_model", "get_vector_store"]

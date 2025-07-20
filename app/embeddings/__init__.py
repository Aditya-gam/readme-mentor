"""Embeddings and vector store functionality for readme-mentor.

This package provides text chunking, embedding generation, and vector store
integration for processing repository content.
"""

from .ingest import ingest_repository

__all__ = ["ingest_repository"]

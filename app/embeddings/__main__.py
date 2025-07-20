"""Command-line interface for the embeddings ingestion pipeline.

This module provides a CLI for ingesting GitHub repositories into the vector store.
It can be run as: python -m app.embeddings <repo_url>
"""

import argparse
import logging
import sys
from pathlib import Path

from ..utils.validators import InvalidRepoURLError, validate_repo_url
from .ingest import ingest_repository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Ingest a GitHub repository for vector search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.embeddings https://github.com/octocat/Hello-World
  python -m app.embeddings https://github.com/user/repo --chunk-size 512
  python -m app.embeddings https://github.com/user/repo --file-pattern "*.md" "docs/**/*.md"
        """,
    )

    parser.add_argument(
        "repo_url",
        help="GitHub repository URL to ingest (e.g., https://github.com/octocat/Hello-World)",
    )

    parser.add_argument(
        "--file-pattern",
        nargs="+",
        default=["README*", "docs/**/*.md"],
        help="File patterns to match (default: README* docs/**/*.md)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Size of each text chunk in characters (default: 1024)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Overlap between chunks in characters (default: 128)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of documents to process in embedding batches (default: 64)",
    )

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformers model to use for embeddings (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--collection-name",
        help="Custom collection name for ChromaDB (default: auto-generated)",
    )

    parser.add_argument(
        "--persist-directory",
        help="Directory to persist ChromaDB data (default: in-memory)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate URL and show what would be processed without actually ingesting",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.persist_directory:
        persist_path = Path(args.persist_directory)
        if persist_path.exists() and not persist_path.is_dir():
            raise ValueError("persist_directory must be a directory")


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration.

    Args:
        verbose: Whether to enable verbose logging
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        setup_logging(args.verbose)

        # Validate arguments
        validate_arguments(args)

        logger.info("Starting repository ingestion CLI")

        # Validate repository URL
        try:
            validated_url = validate_repo_url(args.repo_url)
            logger.info(f"Validated repository URL: {validated_url}")
        except (ValueError, InvalidRepoURLError) as e:
            logger.error(f"Invalid repository URL: {e}")
            return 1

        # Dry run mode
        if args.dry_run:
            logger.info("DRY RUN MODE - No actual ingestion will be performed")
            logger.info(f"Repository URL: {validated_url}")
            logger.info(f"File patterns: {args.file_pattern}")
            logger.info(f"Chunk size: {args.chunk_size}")
            logger.info(f"Chunk overlap: {args.chunk_overlap}")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Embedding model: {args.embedding_model}")
            if args.collection_name:
                logger.info(f"Collection name: {args.collection_name}")
            if args.persist_directory:
                logger.info(f"Persist directory: {args.persist_directory}")
            logger.info("Dry run completed successfully")
            return 0

        # Perform ingestion
        logger.info("Starting repository ingestion...")
        vectorstore = ingest_repository(
            repo_url=validated_url,
            file_glob=tuple(args.file_pattern),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            embedding_model_name=args.embedding_model,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory,
        )

        logger.info("Repository ingestion completed successfully")
        logger.info(
            f"Vector store created with collection: {vectorstore._collection.name}"
        )

        # Show some basic stats if available
        try:
            collection_count = vectorstore._collection.count()
            logger.info(f"Total documents in collection: {collection_count}")
        except Exception as e:
            logger.debug(f"Could not get collection count: {e}")

        return 0

    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        if args.verbose:
            import traceback

            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

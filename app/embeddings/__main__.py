"""Command-line interface for the embeddings ingestion pipeline.

This module provides a CLI for ingesting GitHub repositories into the vector store.
It can be run as: python -m app.embeddings <repo_url>
"""

import argparse
import logging
import sys

from .ingest import ingest_repository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest a GitHub repository for vector search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.embeddings https://github.com/octocat/Hello-World
  python -m app.embeddings https://github.com/user/repo --chunk-size 512
  python -m app.embeddings https://github.com/user/repo --file-glob "*.md" "docs/**/*.md"
  python -m app.embeddings https://github.com/user/repo --persist-dir ./data/chroma
        """,
    )

    parser.add_argument(
        "repo_url",
        help="GitHub repository URL to ingest",
    )

    parser.add_argument(
        "--file-glob",
        nargs="+",
        default=None,
        help="File glob patterns to process (default: README* and docs/**/*.md)",
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
        help="Sentence-transformers model to use for embeddings (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--collection-name",
        default=None,
        help="Custom collection name for ChromaDB (default: auto-generated)",
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Directory to persist ChromaDB data (default: in-memory)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        logger.info(f"Starting ingestion for repository: {args.repo_url}")

        # Run ingestion
        vectorstore = ingest_repository(
            repo_url=args.repo_url,
            file_glob=tuple(args.file_glob) if args.file_glob else None,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            embedding_model_name=args.embedding_model,
            collection_name=args.collection_name,
            persist_directory=args.persist_dir,
        )

        # Print success information
        print("✅ Ingestion completed successfully!")
        print(f"Collection name: {vectorstore._collection.name}")

        if args.persist_dir:
            print(f"Data persisted to: {args.persist_dir}")
        else:
            print("Data stored in memory (not persisted)")

        # Test search functionality
        print("Testing search functionality...")
        results = vectorstore.similarity_search("test", k=1)
        if results:
            print(f"✅ Search test successful - found {len(results)} result(s)")
        else:
            print("⚠️  Search test returned no results")

        return 0

    except KeyboardInterrupt:
        print("Ingestion interrupted by user")
        return 1

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

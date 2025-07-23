"""Enhanced CLI for README-Mentor.

This module provides a user-friendly command-line interface for ingesting
GitHub repositories and starting interactive QA sessions with chat history.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .embeddings.ingest import ingest_repository

# Constants
FULL_TRACEBACK_MSG = "Full traceback:"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_ingest_arguments() -> argparse.Namespace:
    """Parse arguments for the ingest command."""
    parser = argparse.ArgumentParser(
        description="Ingest a GitHub repository for Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  readme-mentor ingest https://github.com/octocat/Hello-World
  readme-mentor ingest https://github.com/user/repo --save
  readme-mentor ingest https://github.com/user/repo --files "*.md" "docs/**/*.md"
  readme-mentor ingest https://github.com/user/repo --fast
        """,
    )

    parser.add_argument(
        "repo_url",
        help="GitHub repository URL to ingest",
    )

    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save data to disk for future use (default: in-memory only)",
    )

    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        help="File patterns to process (default: README* and docs/**/*.md)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (smaller chunks, faster model)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    # Parse arguments starting from the third element (after script name and command)
    return parser.parse_args(sys.argv[2:])


def parse_qa_arguments() -> argparse.Namespace:
    """Parse arguments for the qa command."""
    parser = argparse.ArgumentParser(
        description="Start interactive Q&A session with automatic ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  readme-mentor qa https://github.com/octocat/Hello-World
  readme-mentor qa --repo-id octocat_Hello-World
  readme-mentor qa https://github.com/user/repo --fast --files "*.md"
  readme-mentor qa https://github.com/user/repo --no-save --verbose
        """,
    )

    # Allow either repo_url or repo_id
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "repo_url",
        nargs="?",
        help="GitHub repository URL to load (will auto-ingest if needed)",
    )
    group.add_argument(
        "--repo-id",
        help="Repository ID (e.g., 'owner_repo') - must be pre-ingested",
    )

    # Ingestion options (only apply when repo_url is provided)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save ingested data to disk (in-memory only)",
    )

    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        help="File patterns to process during ingestion (default: README* and docs/**/*.md)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings during ingestion (smaller chunks, faster model)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear chat history at the start of the session",
    )

    # Parse arguments starting from the third element (after script name and command)
    return parser.parse_args(sys.argv[2:])


def parse_arguments() -> argparse.Namespace:
    """Parse main command line arguments."""
    if len(sys.argv) < 2:
        parser = argparse.ArgumentParser(
            description="README-Mentor - AI-powered Q&A over repository documentation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Commands:
  ingest    Ingest a GitHub repository for Q&A
  qa        Start interactive Q&A session (auto-ingests if needed)

Examples:
  readme-mentor ingest https://github.com/octocat/Hello-World
  readme-mentor qa https://github.com/octocat/Hello-World
        """,
        )
        parser.print_help()
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        return parse_ingest_arguments()
    elif command == "qa":
        return parse_qa_arguments()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: ingest, qa")
        sys.exit(1)


def get_ingest_settings(args: argparse.Namespace) -> dict:
    """Get ingestion settings based on user preferences."""
    if args.fast:
        return {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "batch_size": 32,
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
    else:
        return {
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "batch_size": 64,
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }


def run_ingest(args: argparse.Namespace) -> int:
    """Run the ingest command."""
    try:
        print(f"🚀 Starting ingestion for: {args.repo_url}")

        # Get settings based on user preferences
        settings = get_ingest_settings(args)

        # Set up persistence directory if requested
        persist_directory = None
        if args.save:
            from .embeddings.ingest import _extract_repo_slug
            from .utils.validators import validate_repo_url

            validated_url = validate_repo_url(args.repo_url)
            repo_slug = _extract_repo_slug(validated_url)
            persist_directory = f"data/{repo_slug}/chroma"
            Path(persist_directory).parent.mkdir(parents=True, exist_ok=True)
            print(f"💾 Data will be saved to: {persist_directory}")

        # Run ingestion
        vectorstore = ingest_repository(
            repo_url=args.repo_url,
            file_glob=tuple(args.files) if args.files else None,
            persist_directory=persist_directory,
            **settings,
        )

        # Success message
        print("✅ Ingestion completed successfully!")
        print(f"📚 Collection: {vectorstore._collection.name}")

        if args.save:
            print(f"💾 Data saved to: {persist_directory}")
            print("💡 You can now use: readme-mentor qa --repo-id <repo_id>")
        else:
            print("💡 Use --save to persist data for future Q&A sessions")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Ingestion interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        if args.verbose:
            logger.exception(FULL_TRACEBACK_MSG)
        return 1


class ChatSession:
    """Manages an interactive chat session with history."""

    def __init__(self, repo_id: str, clear_history: bool = False):
        self.repo_id = repo_id
        self.chat_history: List[Tuple[str, str]] = []
        self.session_start = datetime.now()

        if clear_history:
            print("🗑️  Chat history cleared")

    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to the history."""
        self.chat_history.append((question, answer))

    def display_history(self):
        """Display the chat history in a formatted way."""
        if not self.chat_history:
            print("💬 No previous messages in this session")
            return

        print(f"\n📜 Chat History ({len(self.chat_history)} exchanges):")
        print("=" * 60)

        for i, (question, answer) in enumerate(self.chat_history, 1):
            print(f"\n💬 Exchange {i}:")
            print(f"❓ Q: {question}")
            print(f"🤖 A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print("-" * 40)

    def get_history_for_backend(self) -> List[Tuple[str, str]]:
        """Get the chat history in the format expected by the backend."""
        return self.chat_history.copy()


def check_repository_exists(repo_id: str) -> bool:
    """Check if a repository has been ingested and is available."""
    try:
        from .embeddings.ingest import get_embedding_model, get_vector_store

        embedding_model = get_embedding_model()
        vectorstore = get_vector_store(repo_id, embedding_model)

        # Test if we can search
        test_results = vectorstore.similarity_search("test", k=1)
        return len(test_results) > 0
    except Exception:
        return False


def auto_ingest_repository(repo_url: str, args: argparse.Namespace) -> str:
    """Automatically ingest a repository and return the repo_id."""
    print("🔍 Repository not found. Starting automatic ingestion...")

    # Get settings based on user preferences
    settings = get_ingest_settings(args)

    # Set up persistence directory (always save for auto-ingestion)
    from .embeddings.ingest import _extract_repo_slug
    from .utils.validators import validate_repo_url

    validated_url = validate_repo_url(repo_url)
    repo_slug = _extract_repo_slug(validated_url)

    persist_directory = None if args.no_save else f"data/{repo_slug}/chroma"
    if persist_directory:
        Path(persist_directory).parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Data will be saved to: {persist_directory}")

    # Run ingestion
    ingest_repository(
        repo_url=repo_url,
        file_glob=tuple(args.files) if args.files else None,
        persist_directory=persist_directory,
        **settings,
    )

    print("✅ Auto-ingestion completed successfully!")
    return repo_slug


def run_qa(args: argparse.Namespace) -> int:
    """Run the interactive Q&A command with enhanced features."""
    try:
        print("🤖 Starting enhanced interactive Q&A session...")

        # Determine repo_id and handle auto-ingestion
        repo_id = None
        if args.repo_id:
            repo_id = args.repo_id
            print(f"📚 Loading pre-ingested repository: {repo_id}")

            if not check_repository_exists(repo_id):
                print(f"❌ Repository '{repo_id}' not found or has no data")
                print(
                    "💡 Try ingesting it first: readme-mentor ingest <repo_url> --save"
                )
                return 1

        elif args.repo_url:
            from .embeddings.ingest import _extract_repo_slug
            from .utils.validators import validate_repo_url

            validated_url = validate_repo_url(args.repo_url)
            repo_slug = _extract_repo_slug(validated_url)
            repo_id = repo_slug

            print(f"📚 Checking repository: {repo_id}")

            # Check if repository exists, if not auto-ingest
            if not check_repository_exists(repo_id):
                repo_id = auto_ingest_repository(args.repo_url, args)

        if not repo_id:
            print("❌ No repository ID or URL provided")
            return 1

        print(f"✅ Repository ready: {repo_id}")

        # Initialize chat session
        chat_session = ChatSession(repo_id, args.clear_history)

        print("\n" + "=" * 60)
        print("💬 Interactive Q&A Session Started")
        print("=" * 60)
        print("💡 Commands:")
        print("   - Type your question and press Enter")
        print("   - Type 'history' to see previous exchanges")
        print("   - Type 'clear' to clear chat history")
        print("   - Type 'quit', 'exit', or 'q' to end session")
        print("   - Type 'help' for this help message")
        print("=" * 60)

        # Interactive Q&A loop
        while True:
            try:
                question = input("\n❓ Question: ").strip()

                # Handle special commands
                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == "history":
                    chat_session.display_history()
                    continue
                elif question.lower() == "clear":
                    chat_session.chat_history.clear()
                    print("🗑️  Chat history cleared")
                    continue
                elif question.lower() == "help":
                    print("\n💡 Available commands:")
                    print("   - Type your question and press Enter")
                    print("   - Type 'history' to see previous exchanges")
                    print("   - Type 'clear' to clear chat history")
                    print("   - Type 'quit', 'exit', or 'q' to end session")
                    print("   - Type 'help' for this help message")
                    continue

                if not question:
                    continue

                print("🤔 Thinking...")

                # Generate answer using the backend with chat history
                from .backend import generate_answer

                result = generate_answer(
                    question, repo_id, history=chat_session.get_history_for_backend()
                )

                answer = result["answer"]

                # Add to chat history
                chat_session.add_exchange(question, answer)

                # Display the answer with formatting
                print("\n🤖 Answer:")
                print(f"{answer}")

                # Display citations if available
                if result.get("citations"):
                    print(f"\n📖 Sources ({len(result['citations'])}):")
                    for i, citation in enumerate(result["citations"], 1):
                        file_path = citation.get("file", "Unknown")
                        start_line = citation.get("start_line", "?")
                        end_line = citation.get("end_line", "?")
                        print(f"  {i}. {file_path} (lines {start_line}-{end_line})")

                # Display performance metrics
                latency = result.get("latency_ms", 0)
                print(f"\n⏱️  Response time: {latency:.0f}ms")
                print(
                    f"💬 Total exchanges in session: {len(chat_session.chat_history)}"
                )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if args.verbose:
                    logger.exception(FULL_TRACEBACK_MSG)

        # Session summary
        session_duration = datetime.now() - chat_session.session_start
        print("\n📊 Session Summary:")
        print(f"   Duration: {session_duration}")
        print(f"   Total exchanges: {len(chat_session.chat_history)}")
        print(f"   Repository: {repo_id}")

        return 0

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Q&A session failed: {e}")
        if args.verbose:
            logger.exception(FULL_TRACEBACK_MSG)
        return 1


def main() -> int:
    """Main CLI entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine command from sys.argv
    command = sys.argv[1] if len(sys.argv) > 1 else None

    # Run the appropriate command
    if command == "ingest":
        return run_ingest(args)
    elif command == "qa":
        return run_qa(args)
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

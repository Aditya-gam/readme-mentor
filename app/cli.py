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
from .logging import UserOutput, setup_logging

# Constants
FULL_TRACEBACK_MSG = "Full traceback:"
CHAT_HISTORY_CLEARED_MSG = "🗑️  Chat history cleared"


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
  readme-mentor ingest https://github.com/user/repo --output-format json
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

    parser.add_argument(
        "--output-format",
        choices=["rich", "plain", "json"],
        default="rich",
        help="Output format (default: rich)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
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
  readme-mentor qa https://github.com/user/repo --output-format json
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

    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save ingested data to disk (default: in-memory only)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Force in-memory only (overrides --save)",
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

    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear chat history at start of session",
    )

    parser.add_argument(
        "--output-format",
        choices=["rich", "plain", "json"],
        default="rich",
        help="Output format (default: rich)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    # Parse arguments starting from the third element (after script name and command)
    return parser.parse_args(sys.argv[2:])


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered README generation and mentoring tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ingest    Ingest a GitHub repository for Q&A
  qa        Start interactive Q&A session

For more help on a command:
  readme-mentor <command> --help
        """,
    )

    parser.add_argument(
        "command",
        choices=["ingest", "qa"],
        help="Command to run",
    )

    # Parse arguments starting from the second element (after script name)
    return parser.parse_args(sys.argv[1:2])


def get_user_output_level(args: argparse.Namespace) -> str:
    """Get user output level based on verbosity flags."""
    if args.quiet:
        return "QUIET"
    elif args.verbose:
        return "DEBUG"
    else:
        return "NORMAL"


def get_ingest_settings(args: argparse.Namespace) -> dict:
    """Get ingestion settings based on user preferences."""
    settings = {}

    if args.fast:
        settings.update(
            {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model_name": "all-MiniLM-L6-v2",  # Faster model
            }
        )

    return settings


def run_ingest(args: argparse.Namespace) -> int:
    """Run the ingest command."""
    # Set up logging based on output format and verbosity
    user_output_level = get_user_output_level(args)
    user_output, dev_logger = setup_logging(
        user_output_level=user_output_level, output_format=args.output_format
    )

    try:
        user_output.start_operation_timer("ingestion")
        user_output.info("🚀 Starting ingestion", emoji="🚀")

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
            user_output.info("💾 Data will be saved", emoji="💾")

        # Run ingestion with progress tracking
        vectorstore = ingest_repository(
            repo_url=args.repo_url,
            file_glob=tuple(args.files) if args.files else None,
            persist_directory=persist_directory,
            user_output=user_output,
            **settings,
        )

        # End timing and get duration
        duration = user_output.end_operation_timer("ingestion")
        user_output.add_performance_metric("ingestion_duration", duration)

        # Get collection info for summary
        collection_name = vectorstore._collection.name

        # Count total files and chunks for summary
        total_files = user_output._performance_metrics.get("total_files", 0)
        total_chunks = user_output._performance_metrics.get("total_chunks", 0)

        # Print comprehensive summary
        user_output.print_ingestion_summary(
            repo_url=args.repo_url,
            total_files=total_files,
            total_chunks=total_chunks,
            duration=duration,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        # Success message
        user_output.success("✅ Ingestion completed successfully!", emoji="✅")

        if args.save:
            user_output.info("💡 You can now use: readme-mentor qa --repo-id <repo_id>")
        else:
            user_output.info("💡 Use --save to persist data for future Q&A sessions")

        # Display performance metrics
        user_output.print_performance_metrics()

        return 0

    except KeyboardInterrupt:
        user_output.error("❌ Ingestion interrupted by user", emoji="❌")
        return 1
    except Exception as e:
        # Enhanced error handling with suggestions
        suggestions = user_output.formatter._get_error_suggestions(e)
        user_output.print_error_summary(
            error=e, context="Repository ingestion", suggestions=suggestions
        )
        if args.verbose:
            dev_logger.exception("Ingestion failed")
        return 1


class ChatSession:
    """Manages an interactive chat session with history."""

    def __init__(
        self, repo_id: str, clear_history: bool = False, user_output: UserOutput = None
    ):
        self.repo_id = repo_id
        self.chat_history: List[Tuple[str, str]] = []
        self.session_start = datetime.now()
        self.user_output = user_output

        if clear_history:
            if user_output:
                user_output.info(CHAT_HISTORY_CLEARED_MSG, emoji="🗑️")
            else:
                print(CHAT_HISTORY_CLEARED_MSG)

    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to the history."""
        self.chat_history.append((question, answer))

    def display_history(self):
        """Display the chat history in a formatted way."""
        if not self.chat_history:
            self._display_empty_history()
            return

        if self.user_output:
            self._display_rich_history()
        else:
            self._display_plain_history()

    def _display_empty_history(self):
        """Display message when no history exists."""
        if self.user_output:
            self.user_output.info("💬 No previous messages in this session", emoji="💬")
        else:
            print("💬 No previous messages in this session")

    def _display_rich_history(self):
        """Display history using rich formatting."""
        self.user_output.info(
            f"📜 Chat History ({len(self.chat_history)} exchanges)", emoji="📜"
        )
        self.user_output.print_separator("=", 60)

        for i, (question, answer) in enumerate(self.chat_history, 1):
            self.user_output.info(f"💬 Exchange {i}:", emoji="💬")
            self.user_output.info(f"❓ Q: {question}")
            self.user_output.info(f"🤖 A: {self._truncate_answer(answer)}")
            self.user_output.print_separator("-", 40)

    def _display_plain_history(self):
        """Display history using plain text formatting."""
        print(f"\n📜 Chat History ({len(self.chat_history)} exchanges):")
        print("=" * 60)

        for i, (question, answer) in enumerate(self.chat_history, 1):
            print(f"\n💬 Exchange {i}:")
            print(f"❓ Q: {question}")
            print(f"🤖 A: {self._truncate_answer(answer)}")
            print("-" * 40)

    def _truncate_answer(self, answer: str) -> str:
        """Truncate answer to 200 characters with ellipsis if needed."""
        return f"{answer[:200]}{'...' if len(answer) > 200 else ''}"

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


def auto_ingest_repository(
    repo_url: str, args: argparse.Namespace, user_output: UserOutput
) -> str:
    """Automatically ingest a repository if it doesn't exist."""
    from .embeddings.ingest import _extract_repo_slug
    from .utils.validators import validate_repo_url

    validated_url = validate_repo_url(repo_url)
    repo_slug = _extract_repo_slug(validated_url)

    if not check_repository_exists(repo_slug):
        user_output.info("📥 Repository not found, auto-ingesting...", emoji="📥")

        # Create ingest args for auto-ingestion
        ingest_args = argparse.Namespace(
            repo_url=validated_url,
            save=args.save,
            files=args.files,
            fast=args.fast,
            verbose=args.verbose,
            output_format=args.output_format,
            quiet=args.quiet,
        )

        # Run ingestion
        result = run_ingest(ingest_args)
        if result != 0:
            raise RuntimeError("Auto-ingestion failed")

        user_output.success("✅ Auto-ingestion completed!", emoji="✅")
    else:
        user_output.info("✅ Repository already ingested", emoji="✅")

    return repo_slug


def _handle_special_commands(question: str, chat_session: ChatSession) -> bool:
    """Handle special commands in the chat session."""
    question_lower = question.lower().strip()

    if question_lower in ["quit", "exit", "q"]:
        return True
    elif question_lower == "history":
        chat_session.display_history()
        return True
    elif question_lower == "clear":
        chat_session.chat_history.clear()
        if chat_session.user_output:
            chat_session.user_output.info(CHAT_HISTORY_CLEARED_MSG, emoji="🗑️")
        else:
            print(CHAT_HISTORY_CLEARED_MSG)
        return True
    elif question_lower == "help":
        _print_session_help()
        return True

    return False


def _display_answer_with_metadata(result: dict, chat_session: ChatSession):
    """Display answer with citations and performance metrics."""
    answer = result["answer"]
    citations = result.get("citations", [])
    metadata = {
        "latency_ms": result.get("latency_ms", 0),
        "total_exchanges": len(chat_session.chat_history) + 1,
    }

    if chat_session.user_output:
        chat_session.user_output.print_qa_session(
            question=chat_session.chat_history[-1][0]
            if chat_session.chat_history
            else "Unknown",
            answer=answer,
            citations=citations,
            metadata=metadata,
        )
    else:
        # Fallback to old format
        print("\n🤖 Answer:")
        print(f"{answer}")

        # Display citations if available
        if citations:
            print(f"\n📖 Sources ({len(citations)}):")
            for i, citation in enumerate(citations, 1):
                file_path = citation.get("file", "Unknown")
                start_line = citation.get("start_line", "?")
                end_line = citation.get("end_line", "?")
                print(f"  {i}. {file_path} (lines {start_line}-{end_line})")

        # Display performance metrics
        latency = result.get("latency_ms", 0)
        print(f"\n⏱️  Response time: {latency:.0f}ms")
        print(f"💬 Total exchanges in session: {len(chat_session.chat_history)}")


def _process_question(question: str, repo_id: str, chat_session: ChatSession):
    """Process a question and display the answer."""
    from .rag.chain import generate_answer

    # Generate answer
    result = generate_answer(question, repo_id, chat_session.get_history_for_backend())

    # Add to chat history
    chat_session.add_exchange(question, result["answer"])

    # Display answer
    _display_answer_with_metadata(result, chat_session)


def _setup_repository(args: argparse.Namespace, user_output: UserOutput) -> str:
    """Set up repository for Q&A session."""
    if args.repo_id:
        # Use existing repository
        if not check_repository_exists(args.repo_id):
            raise FileNotFoundError(
                f"Repository '{args.repo_id}' not found. Please ingest it first."
            )
        user_output.info(f"✅ Using existing repository: {args.repo_id}")
        return args.repo_id
    else:
        # Auto-ingest repository
        return auto_ingest_repository(args.repo_url, args, user_output)


def _print_session_help():
    """Print session help information."""
    help_text = """
🤖 README-Mentor Q&A Session Help
================================

Commands:
   - Type 'history' to see previous exchanges
   - Type 'clear' to clear chat history
   - Type 'quit', 'exit', or 'q' to end session
   - Type 'help' for this help message

Tips:
   - Ask specific questions about the repository
   - Use natural language to describe what you want to know
   - The AI will search through the repository content to find relevant information
"""
    print(help_text)


def _run_interactive_loop(
    repo_id: str, chat_session: ChatSession, args: argparse.Namespace
):
    """Run the interactive Q&A loop."""
    while True:
        try:
            question = input("\n❓ Question: ").strip()

            if _should_exit_loop(question, chat_session):
                break

            if not question:
                continue

            # Process the question
            _process_question(question, repo_id, chat_session)

        except KeyboardInterrupt:
            _handle_keyboard_interrupt(chat_session)
            break
        except Exception as e:
            _handle_processing_error(e, chat_session, args)


def _should_exit_loop(question: str, chat_session: ChatSession) -> bool:
    """Check if the loop should exit based on the question."""
    if _handle_special_commands(question, chat_session):
        return question.lower() in ["quit", "exit", "q"]
    return False


def _handle_keyboard_interrupt(chat_session: ChatSession):
    """Handle keyboard interrupt gracefully."""
    if chat_session.user_output:
        chat_session.user_output.info("👋 Goodbye!", emoji="👋")
    else:
        print("\n👋 Goodbye!")


def _handle_processing_error(
    e: Exception, chat_session: ChatSession, args: argparse.Namespace
):
    """Handle errors during question processing."""
    if chat_session.user_output:
        chat_session.user_output.error("❌ Error processing question", error=e)
    else:
        print(f"❌ Error: {e}")

    if args.verbose:
        logging.getLogger().exception("Error in interactive loop")


def _print_session_summary(
    chat_session: ChatSession, repo_id: str, user_output: UserOutput = None
):
    """Print session summary."""
    session_duration = datetime.now() - chat_session.session_start

    if user_output:
        user_output.info("📊 Session Summary", emoji="📊")
        user_output.info(f"   Duration: {session_duration}")
        user_output.info(f"   Total exchanges: {len(chat_session.chat_history)}")
        user_output.info(f"   Repository: {repo_id}")
    else:
        print("\n📊 Session Summary:")
        print(f"   Duration: {session_duration}")
        print(f"   Total exchanges: {len(chat_session.chat_history)}")
        print(f"   Repository: {repo_id}")


def run_qa(args: argparse.Namespace) -> int:
    """Run the interactive Q&A command with enhanced features."""
    # Set up logging based on output format and verbosity
    user_output_level = get_user_output_level(args)
    user_output, dev_logger = setup_logging(
        user_output_level=user_output_level, output_format=args.output_format
    )

    try:
        user_output.start_operation_timer("qa_session")
        user_output.info("🤖 Starting enhanced interactive Q&A session...", emoji="🤖")

        # Set up repository
        repo_id = _setup_repository(args, user_output)
        user_output.success(f"✅ Repository ready: {repo_id}", emoji="✅")

        # Initialize chat session
        chat_session = ChatSession(repo_id, args.clear_history, user_output)
        _print_session_help()

        # Run interactive loop
        _run_interactive_loop(repo_id, chat_session, args)

        # End timing and get duration
        duration = user_output.end_operation_timer("qa_session")
        user_output.add_performance_metric("qa_session_duration", duration)

        # Print session summary
        _print_session_summary(chat_session, repo_id, user_output)

        # Display performance metrics
        user_output.print_performance_metrics()

        return 0

    except KeyboardInterrupt:
        user_output.info("👋 Goodbye!", emoji="👋")
        return 0
    except Exception as e:
        # Enhanced error handling with suggestions
        suggestions = user_output.formatter._get_error_suggestions(e)
        user_output.print_error_summary(
            error=e, context="Interactive Q&A session", suggestions=suggestions
        )
        if args.verbose:
            dev_logger.exception("Q&A session failed")
        return 1


def main() -> int:
    """Main CLI entry point."""
    # Determine command from sys.argv
    command = sys.argv[1] if len(sys.argv) > 1 else None

    # Run the appropriate command
    if command == "ingest":
        ingest_args = parse_ingest_arguments()
        return run_ingest(ingest_args)
    elif command == "qa":
        qa_args = parse_qa_arguments()
        return run_qa(qa_args)
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

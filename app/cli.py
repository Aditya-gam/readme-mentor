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
from .errors import handle_exception
from .logging import UserOutput, setup_logging
from .logging.enums import VerbosityLevel

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
  readme-mentor ingest https://github.com/user/repo --verbosity 2
  readme-mentor ingest https://github.com/user/repo --quiet
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

    # Enhanced verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbosity",
        "-V",
        type=int,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)",
    )
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (equivalent to --verbosity 2)",
    )
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors (equivalent to --verbosity 0)",
    )

    parser.add_argument(
        "--output-format",
        choices=["rich", "plain", "json"],
        default="rich",
        help="Output format (default: rich)",
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
  readme-mentor qa https://github.com/user/repo --verbosity 3
  readme-mentor qa https://github.com/user/repo --quiet
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

    # Enhanced verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbosity",
        "-V",
        type=int,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)",
    )
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (equivalent to --verbosity 2)",
    )
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors (equivalent to --verbosity 0)",
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

Verbosity Levels:
  0 (quiet)    Only critical errors and final results
  1 (normal)   Success/failure status, basic metrics, progress indicators (default)
  2 (verbose)  Detailed operation steps, extended metrics, configuration details
  3 (debug)    All available information, raw data, internal state, full analysis

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
    """Get user output level based on verbosity flags.

    Args:
        args: Parsed command line arguments

    Returns:
        String representation of verbosity level
    """
    # Check for explicit verbosity level first
    if hasattr(args, "verbosity") and args.verbosity is not None:
        return VerbosityLevel(args.verbosity).to_string()

    # Check for legacy flags
    if hasattr(args, "quiet") and args.quiet:
        return VerbosityLevel.QUIET.to_string()
    elif hasattr(args, "verbose") and args.verbose:
        return VerbosityLevel.VERBOSE.to_string()

    # Default to normal level
    return VerbosityLevel.NORMAL.to_string()


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

        # Display configuration details if verbose
        if user_output.config.show_configuration_details:
            user_output.config_detail("Repository URL", args.repo_url)
            user_output.config_detail("Save to disk", args.save)
            user_output.config_detail("File patterns", args.files or "default")
            user_output.config_detail("Fast mode", args.fast)

        # Get settings based on user preferences
        settings = get_ingest_settings(args)

        # Display settings if verbose
        if user_output.config.show_configuration_details:
            for key, value in settings.items():
                user_output.config_detail(f"Setting: {key}", value)

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

            if user_output.config.show_configuration_details:
                user_output.config_detail("Persistence directory", persist_directory)

        # Run ingestion with progress tracking
        user_output.step("Starting repository ingestion", operation="ingestion")
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

        # Print operation summary if verbose
        user_output.print_operation_summary("ingestion")

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
        # Enhanced error handling with new error system
        user_error = handle_exception(
            e,
            context={
                "repo_url": args.repo_url,
                "operation": "repository_ingestion",
                "component": "cli",
            },
            operation="repository_ingestion",
        )
        user_output.formatter.operation_error("Repository Ingestion", user_error)

        # Log error details if verbose
        if user_output.config.show_error_details:
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
            verbosity=getattr(args, "verbosity", None),
            quiet=getattr(args, "quiet", False),
            output_format=args.output_format,
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

    user_output = chat_session.user_output

    # Start timing the question processing
    if user_output:
        user_output.start_operation_timer("question_processing")
        user_output.step("Processing question", operation="question_processing")

    # Generate answer
    result = generate_answer(question, repo_id, chat_session.get_history_for_backend())

    # Add performance metrics
    if user_output:
        latency = result.get("latency_ms", 0)
        user_output.add_performance_metric("question_latency_ms", latency)
        user_output.add_token_count("question_processing", result.get("token_count", 0))

        # End timing
        user_output.end_operation_timer("question_processing")

        # Print operation summary if verbose
        user_output.print_operation_summary("question_processing")

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

        # Display configuration details if verbose
        if user_output.config.show_configuration_details:
            user_output.config_detail("Repository ID", getattr(args, "repo_id", "auto"))
            user_output.config_detail(
                "Repository URL", getattr(args, "repo_url", "none")
            )
            user_output.config_detail("Clear history", args.clear_history)
            user_output.config_detail("Output format", args.output_format)

        # Set up repository
        user_output.step("Setting up repository", operation="qa_session")
        repo_id = _setup_repository(args, user_output)
        user_output.success(f"✅ Repository ready: {repo_id}", emoji="✅")

        # Initialize chat session
        user_output.step("Initializing chat session", operation="qa_session")
        chat_session = ChatSession(repo_id, args.clear_history, user_output)
        _print_session_help()

        # Run interactive loop
        user_output.step("Starting interactive loop", operation="qa_session")
        _run_interactive_loop(repo_id, chat_session, args)

        # End timing and get duration
        duration = user_output.end_operation_timer("qa_session")
        user_output.add_performance_metric("qa_session_duration", duration)

        # Print session summary
        _print_session_summary(chat_session, repo_id, user_output)

        # Print operation summary if verbose
        user_output.print_operation_summary("qa_session")

        # Display performance metrics
        user_output.print_performance_metrics()

        return 0

    except KeyboardInterrupt:
        user_output.info("👋 Goodbye!", emoji="👋")
        return 0
    except Exception as e:
        # Enhanced error handling with new error system
        user_error = handle_exception(
            e,
            context={"operation": "interactive_qa", "component": "cli"},
            operation="interactive_qa",
        )
        user_output.formatter.operation_error("Interactive Q&A Session", user_error)

        # Log error details if verbose
        if user_output.config.show_error_details:
            dev_logger.exception("Q&A session failed")
        return 1


def main() -> int:
    """Main CLI entry point."""
    # Determine command from sys.argv
    command = sys.argv[1] if len(sys.argv) > 1 else None

    # Check if this is a subcommand help request
    if (
        command in ["ingest", "qa"]
        and len(sys.argv) > 2
        and sys.argv[2] in ["--help", "-h"]
    ):
        if command == "ingest":
            # Create a temporary parser to show help
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
  readme-mentor ingest https://github.com/user/repo --verbosity 2
  readme-mentor ingest https://github.com/user/repo --quiet
                """,
            )
            parser.add_argument("repo_url", help="GitHub repository URL to ingest")
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
            verbosity_group = parser.add_mutually_exclusive_group()
            verbosity_group.add_argument(
                "--verbosity",
                "-V",
                type=int,
                choices=[0, 1, 2, 3],
                help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)",
            )
            verbosity_group.add_argument(
                "--verbose",
                "-v",
                action="store_true",
                help="Enable verbose output (equivalent to --verbosity 2)",
            )
            verbosity_group.add_argument(
                "--quiet",
                "-q",
                action="store_true",
                help="Suppress all output except errors (equivalent to --verbosity 0)",
            )
            parser.add_argument(
                "--output-format",
                choices=["rich", "plain", "json"],
                default="rich",
                help="Output format (default: rich)",
            )
            parser.print_help()
        elif command == "qa":
            # Create a temporary parser to show help
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
  readme-mentor qa https://github.com/user/repo --verbosity 3
  readme-mentor qa https://github.com/user/repo --quiet
                """,
            )
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument(
                "repo_url",
                nargs="?",
                help="GitHub repository URL to load (will auto-ingest if needed)",
            )
            group.add_argument(
                "--repo-id", help="Repository ID for existing ingested repository"
            )
            parser.add_argument(
                "--clear-history",
                action="store_true",
                help="Clear chat history at start",
            )
            parser.add_argument(
                "--save",
                "-s",
                action="store_true",
                help="Save data to disk for future use",
            )
            parser.add_argument(
                "--no-save",
                action="store_true",
                help="Don't save data to disk (default)",
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
            verbosity_group = parser.add_mutually_exclusive_group()
            verbosity_group.add_argument(
                "--verbosity",
                "-V",
                type=int,
                choices=[0, 1, 2, 3],
                help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)",
            )
            verbosity_group.add_argument(
                "--verbose",
                "-v",
                action="store_true",
                help="Enable verbose output (equivalent to --verbosity 2)",
            )
            verbosity_group.add_argument(
                "--quiet",
                "-q",
                action="store_true",
                help="Suppress all output except errors (equivalent to --verbosity 0)",
            )
            parser.add_argument(
                "--output-format",
                choices=["rich", "plain", "json"],
                default="rich",
                help="Output format (default: rich)",
            )
            parser.print_help()
        return 0

    # Check if help is requested for main command
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        # Show main help
        parser = argparse.ArgumentParser(
            description="AI-powered README generation and mentoring tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Commands:
  ingest    Ingest a GitHub repository for Q&A
  qa        Start interactive Q&A session

Verbosity Levels:
  0 (quiet)    Only critical errors and final results
  1 (normal)   Success/failure status, basic metrics, progress indicators (default)
  2 (verbose)  Detailed operation steps, extended metrics, configuration details
  3 (debug)    All available information, raw data, internal state, full analysis

For more help on a command:
  readme-mentor <command> --help
        """,
        )
        parser.add_argument(
            "command",
            choices=["ingest", "qa"],
            help="Command to run",
        )
        parser.print_help()
        return 0

    # Run the appropriate command
    if command == "ingest":
        ingest_args = parse_ingest_arguments()
        return run_ingest(ingest_args)
    elif command == "qa":
        qa_args = parse_qa_arguments()
        return run_qa(qa_args)
    else:
        print(f"Unknown command: {command}")
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())

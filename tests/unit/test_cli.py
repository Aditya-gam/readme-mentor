"""Tests for CLI functionality and argument parsing."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from app.cli import (
    ChatSession,
    _display_answer_with_metadata,
    _handle_special_commands,
    _print_session_help,
    _print_session_summary,
    _process_question,
    _run_interactive_loop,
    _setup_repository,
    auto_ingest_repository,
    check_repository_exists,
    get_ingest_settings,
    main,
    parse_arguments,
    parse_ingest_arguments,
    parse_qa_arguments,
    run_ingest,
    run_qa,
)


class TestArgumentParsing:
    """Test argument parsing functions."""

    @patch("sys.argv", ["readme-mentor", "ingest", "https://github.com/test/repo"])
    def test_parse_ingest_arguments_basic(self):
        """Test basic ingest argument parsing."""
        args = parse_ingest_arguments()
        assert args.repo_url == "https://github.com/test/repo"
        assert not args.save
        assert not args.files
        assert not args.fast
        assert not args.verbose

    @patch(
        "sys.argv",
        [
            "readme-mentor",
            "ingest",
            "https://github.com/test/repo",
            "--save",
            "--fast",
            "--verbose",
        ],
    )
    def test_parse_ingest_arguments_with_flags(self):
        """Test ingest argument parsing with all flags."""
        args = parse_ingest_arguments()
        assert args.repo_url == "https://github.com/test/repo"
        assert args.save
        assert args.fast
        assert args.verbose

    @patch(
        "sys.argv",
        [
            "readme-mentor",
            "ingest",
            "https://github.com/test/repo",
            "--files",
            "*.md",
            "docs/**/*.md",
        ],
    )
    def test_parse_ingest_arguments_with_files(self):
        """Test ingest argument parsing with file patterns."""
        args = parse_ingest_arguments()
        assert args.repo_url == "https://github.com/test/repo"
        assert args.files == ["*.md", "docs/**/*.md"]

    @patch("sys.argv", ["readme-mentor", "qa", "https://github.com/test/repo"])
    def test_parse_qa_arguments_with_url(self):
        """Test QA argument parsing with repository URL."""
        args = parse_qa_arguments()
        assert args.repo_url == "https://github.com/test/repo"
        assert not args.repo_id
        assert not args.no_save
        assert not args.files
        assert not args.fast
        assert not args.verbose
        assert not args.clear_history

    @patch("sys.argv", ["readme-mentor", "qa", "--repo-id", "test_repo"])
    def test_parse_qa_arguments_with_repo_id(self):
        """Test QA argument parsing with repository ID."""
        args = parse_qa_arguments()
        assert not args.repo_url
        assert args.repo_id == "test_repo"

    @patch(
        "sys.argv",
        [
            "readme-mentor",
            "qa",
            "https://github.com/test/repo",
            "--fast",
            "--verbose",
            "--clear-history",
        ],
    )
    def test_parse_qa_arguments_with_flags(self):
        """Test QA argument parsing with all flags."""
        args = parse_qa_arguments()
        assert args.repo_url == "https://github.com/test/repo"
        assert args.fast
        assert args.verbose
        assert args.clear_history

    @patch("sys.argv", ["readme-mentor"])
    def test_parse_arguments_no_command(self):
        """Test argument parsing with no command (should exit)."""
        with pytest.raises(SystemExit):
            parse_arguments()

    @patch("sys.argv", ["readme-mentor", "ingest", "https://github.com/test/repo"])
    def test_parse_arguments_ingest(self):
        """Test main argument parsing for ingest command."""
        args = parse_arguments()
        assert args.repo_url == "https://github.com/test/repo"

    @patch("sys.argv", ["readme-mentor", "qa", "https://github.com/test/repo"])
    def test_parse_arguments_qa(self):
        """Test main argument parsing for qa command."""
        args = parse_arguments()
        assert args.repo_url == "https://github.com/test/repo"

    @patch("sys.argv", ["readme-mentor", "unknown"])
    def test_parse_arguments_unknown_command(self):
        """Test argument parsing with unknown command (should exit)."""
        with pytest.raises(SystemExit):
            parse_arguments()


class TestIngestSettings:
    """Test ingest settings configuration."""

    def test_get_ingest_settings_fast(self):
        """Test fast ingest settings."""
        args = Mock(fast=True)
        settings = get_ingest_settings(args)
        assert settings["chunk_size"] == 512
        assert settings["chunk_overlap"] == 64
        assert settings["batch_size"] == 32

    def test_get_ingest_settings_normal(self):
        """Test normal ingest settings."""
        args = Mock(fast=False)
        settings = get_ingest_settings(args)
        assert settings["chunk_size"] == 1024
        assert settings["chunk_overlap"] == 128
        assert settings["batch_size"] == 64


class TestChatSession:
    """Test ChatSession class functionality."""

    def test_chat_session_initialization(self):
        """Test ChatSession initialization."""
        session = ChatSession("test_repo")
        assert session.repo_id == "test_repo"
        assert session.chat_history == []
        assert isinstance(session.session_start, datetime)

    def test_chat_session_clear_history(self):
        """Test ChatSession with clear history flag."""
        session = ChatSession("test_repo", clear_history=True)
        assert session.repo_id == "test_repo"
        assert session.chat_history == []

    def test_add_exchange(self):
        """Test adding question-answer exchanges."""
        session = ChatSession("test_repo")
        session.add_exchange("What is this?", "This is a test.")
        assert len(session.chat_history) == 1
        assert session.chat_history[0] == ("What is this?", "This is a test.")

    def test_get_history_for_backend(self):
        """Test getting history for backend."""
        session = ChatSession("test_repo")
        session.add_exchange("Q1", "A1")
        session.add_exchange("Q2", "A2")
        history = session.get_history_for_backend()
        assert history == [("Q1", "A1"), ("Q2", "A2")]
        # Ensure it returns a copy
        assert history is not session.chat_history

    @patch("builtins.print")
    def test_display_history_empty(self, mock_print):
        """Test displaying empty chat history."""
        session = ChatSession("test_repo")
        session.display_history()
        mock_print.assert_called_with("💬 No previous messages in this session")

    @patch("builtins.print")
    def test_display_history_with_exchanges(self, mock_print):
        """Test displaying chat history with exchanges."""
        session = ChatSession("test_repo")
        session.add_exchange("Short Q", "Short A")
        session.add_exchange(
            "Long question that should be truncated",
            "Long answer that should be truncated",
        )
        session.display_history()
        # Should call print multiple times for the history display
        assert mock_print.call_count > 1


class TestHelperFunctions:
    """Test CLI helper functions."""

    def test_handle_special_commands_quit(self):
        """Test handling quit command."""
        session = ChatSession("test_repo")
        with patch("builtins.print") as mock_print:
            result = _handle_special_commands("quit", session)
            assert result is True
            mock_print.assert_called_with("👋 Goodbye!")

    def test_handle_special_commands_history(self):
        """Test handling history command."""
        session = ChatSession("test_repo")
        session.add_exchange("Q", "A")
        with patch("builtins.print"):
            result = _handle_special_commands("history", session)
            assert result is True

    def test_handle_special_commands_clear(self):
        """Test handling clear command."""
        session = ChatSession("test_repo")
        session.add_exchange("Q", "A")
        with patch("builtins.print") as mock_print:
            result = _handle_special_commands("clear", session)
            assert result is True
            assert len(session.chat_history) == 0
            mock_print.assert_called_with("🗑️  Chat history cleared")

    def test_handle_special_commands_help(self):
        """Test handling help command."""
        session = ChatSession("test_repo")
        with patch("builtins.print") as mock_print:
            result = _handle_special_commands("help", session)
            assert result is True
            assert mock_print.call_count > 1

    def test_handle_special_commands_normal_question(self):
        """Test handling normal question (not a special command)."""
        session = ChatSession("test_repo")
        result = _handle_special_commands("What is this?", session)
        assert result is False

    @patch("builtins.print")
    def test_display_answer_with_metadata(self, mock_print):
        """Test displaying answer with metadata."""
        session = ChatSession("test_repo")
        result = {
            "answer": "Test answer",
            "citations": [{"file": "test.md", "start_line": 1, "end_line": 5}],
            "latency_ms": 150.5,
        }
        _display_answer_with_metadata(result, session)
        # Should print answer, citations, and metrics
        assert mock_print.call_count >= 3

    @patch("builtins.print")
    def test_display_answer_without_citations(self, mock_print):
        """Test displaying answer without citations."""
        session = ChatSession("test_repo")
        result = {"answer": "Test answer", "latency_ms": 100}
        _display_answer_with_metadata(result, session)
        # Should print answer and metrics but not citations
        assert mock_print.call_count >= 2

    @patch("app.backend.generate_answer")
    @patch("builtins.print")
    def test_process_question(self, mock_print, mock_generate):
        """Test processing a question."""
        session = ChatSession("test_repo")
        mock_generate.return_value = {"answer": "Test answer", "latency_ms": 100}
        _process_question("Test question", "test_repo", session)
        mock_generate.assert_called_once()
        assert len(session.chat_history) == 1
        assert session.chat_history[0][0] == "Test question"

    @patch("app.cli.check_repository_exists")
    def test_setup_repository_with_repo_id_exists(self, mock_check):
        """Test setting up repository with existing repo_id."""
        mock_check.return_value = True
        args = Mock(repo_id="test_repo", repo_url=None)
        with patch("builtins.print") as mock_print:
            repo_id = _setup_repository(args)
            assert repo_id == "test_repo"
            mock_print.assert_called_with(
                "📚 Loading pre-ingested repository: test_repo"
            )

    @patch("app.cli.check_repository_exists")
    def test_setup_repository_with_repo_id_not_exists(self, mock_check):
        """Test setting up repository with non-existent repo_id."""
        mock_check.return_value = False
        args = Mock(repo_id="test_repo", repo_url=None)
        with pytest.raises(ValueError, match="Repository 'test_repo' not found"):
            _setup_repository(args)

    @patch("app.cli.check_repository_exists")
    @patch("app.cli.auto_ingest_repository")
    @patch("app.utils.validators.validate_repo_url")
    @patch("app.embeddings.ingest._extract_repo_slug")
    def test_setup_repository_with_url_auto_ingest(
        self, mock_extract, mock_validate, mock_auto_ingest, mock_check
    ):
        """Test setting up repository with URL that needs auto-ingestion."""
        mock_validate.return_value = "https://github.com/test/repo"
        mock_extract.return_value = "test_repo"
        mock_check.return_value = False
        mock_auto_ingest.return_value = "test_repo"
        args = Mock(repo_id=None, repo_url="https://github.com/test/repo")
        with patch("builtins.print"):
            repo_id = _setup_repository(args)
            assert repo_id == "test_repo"
            mock_auto_ingest.assert_called_once()

    def test_setup_repository_no_args(self):
        """Test setting up repository with no valid arguments."""
        args = Mock(repo_id=None, repo_url=None)
        with pytest.raises(ValueError, match="No repository ID or URL provided"):
            _setup_repository(args)

    @patch("builtins.print")
    def test_print_session_help(self, mock_print):
        """Test printing session help."""
        _print_session_help()
        assert mock_print.call_count >= 6  # Multiple lines of help text

    @patch("builtins.print")
    def test_print_session_summary(self, mock_print):
        """Test printing session summary."""
        session = ChatSession("test_repo")
        session.add_exchange("Q1", "A1")
        session.add_exchange("Q2", "A2")
        _print_session_summary(session, "test_repo")
        assert mock_print.call_count >= 3  # Summary lines


class TestRepositoryFunctions:
    """Test repository-related functions."""

    @patch("app.embeddings.ingest.get_embedding_model")
    @patch("app.embeddings.ingest.get_vector_store")
    def test_check_repository_exists_true(self, mock_get_store, mock_get_model):
        """Test checking if repository exists (returns True)."""
        mock_model = Mock()
        mock_store = Mock()
        mock_store.similarity_search.return_value = [Mock()]
        mock_get_model.return_value = mock_model
        mock_get_store.return_value = mock_store

        result = check_repository_exists("test_repo")
        assert result is True

    @patch("app.embeddings.ingest.get_embedding_model")
    @patch("app.embeddings.ingest.get_vector_store")
    def test_check_repository_exists_false(self, mock_get_store, mock_get_model):
        """Test checking if repository exists (returns False)."""
        mock_model = Mock()
        mock_store = Mock()
        mock_store.similarity_search.side_effect = Exception("Not found")
        mock_get_model.return_value = mock_model
        mock_get_store.return_value = mock_store

        result = check_repository_exists("test_repo")
        assert result is False

    @patch("app.cli.ingest_repository")
    @patch("app.utils.validators.validate_repo_url")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.print")
    def test_auto_ingest_repository(
        self, mock_print, mock_mkdir, mock_extract, mock_validate, mock_ingest
    ):
        """Test auto-ingesting a repository."""
        mock_validate.return_value = "https://github.com/test/repo"
        mock_extract.return_value = "test_repo"
        mock_ingest.return_value = Mock()
        args = Mock(no_save=False, files=None, fast=False)

        result = auto_ingest_repository("https://github.com/test/repo", args)
        assert result == "test_repo"
        mock_ingest.assert_called_once()


class TestMainFunctions:
    """Test main CLI functions."""

    @patch("app.cli.run_ingest")
    @patch("app.cli.parse_arguments")
    def test_main_ingest(self, mock_parse, mock_run_ingest):
        """Test main function with ingest command."""
        mock_args = Mock()
        mock_parse.return_value = mock_args
        mock_run_ingest.return_value = 0

        with patch(
            "sys.argv", ["readme-mentor", "ingest", "https://github.com/test/repo"]
        ):
            result = main()
            assert result == 0
            mock_run_ingest.assert_called_once_with(mock_args)

    @patch("app.cli.run_qa")
    @patch("app.cli.parse_arguments")
    def test_main_qa(self, mock_parse, mock_run_qa):
        """Test main function with qa command."""
        mock_args = Mock()
        mock_parse.return_value = mock_args
        mock_run_qa.return_value = 0

        with patch("sys.argv", ["readme-mentor", "qa", "https://github.com/test/repo"]):
            result = main()
            assert result == 0
            mock_run_qa.assert_called_once_with(mock_args)

    @patch("app.cli.parse_arguments")
    def test_main_unknown_command(self, mock_parse):
        """Test main function with unknown command."""
        mock_parse.side_effect = SystemExit(1)

        with patch("sys.argv", ["readme-mentor", "unknown"]):
            with pytest.raises(SystemExit):
                main()

    @patch("app.cli.ingest_repository")
    @patch("app.utils.validators.validate_repo_url")
    @patch("app.embeddings.ingest._extract_repo_slug")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.print")
    def test_run_ingest_success(
        self, mock_print, mock_mkdir, mock_extract, mock_validate, mock_ingest
    ):
        """Test successful ingest run."""
        mock_validate.return_value = "https://github.com/test/repo"
        mock_extract.return_value = "test_repo"
        mock_vectorstore = Mock()
        mock_vectorstore._collection.name = "test_collection"
        mock_ingest.return_value = mock_vectorstore

        args = Mock(
            repo_url="https://github.com/test/repo",
            save=False,
            files=None,
            fast=False,
            verbose=False,
        )

        result = run_ingest(args)
        assert result == 0
        mock_ingest.assert_called_once()

    @patch("builtins.print")
    def test_run_ingest_keyboard_interrupt(self, mock_print):
        """Test ingest run with keyboard interrupt."""
        args = Mock(
            repo_url="https://github.com/test/repo",
            save=False,
            files=None,
            fast=False,
            verbose=False,
        )

        with patch("app.cli.ingest_repository", side_effect=KeyboardInterrupt):
            result = run_ingest(args)
            assert result == 1
            mock_print.assert_called_with("\n❌ Ingestion interrupted by user")

    @patch("builtins.print")
    def test_run_ingest_exception(self, mock_print):
        """Test ingest run with exception."""
        args = Mock(
            repo_url="https://github.com/test/repo",
            save=False,
            files=None,
            fast=False,
            verbose=False,
        )

        with patch("app.cli.ingest_repository", side_effect=Exception("Test error")):
            result = run_ingest(args)
            assert result == 1
            mock_print.assert_called_with("❌ Ingestion failed: Test error")

    @patch("app.cli._setup_repository")
    @patch("app.cli._run_interactive_loop")
    @patch("app.cli._print_session_summary")
    @patch("builtins.print")
    def test_run_qa_success(self, mock_print, mock_summary, mock_loop, mock_setup):
        """Test successful QA run."""
        mock_setup.return_value = "test_repo"

        args = Mock(clear_history=False, verbose=False)

        result = run_qa(args)
        assert result == 0
        mock_setup.assert_called_once_with(args)
        mock_loop.assert_called_once()
        mock_summary.assert_called_once()

    @patch("builtins.print")
    def test_run_qa_keyboard_interrupt(self, mock_print):
        """Test QA run with keyboard interrupt."""
        args = Mock()

        with patch("app.cli._setup_repository", side_effect=KeyboardInterrupt):
            result = run_qa(args)
            assert result == 0
            mock_print.assert_called_with("\n👋 Goodbye!")

    @patch("builtins.print")
    def test_run_qa_exception(self, mock_print):
        """Test QA run with exception."""
        args = Mock(verbose=False)

        with patch("app.cli._setup_repository", side_effect=Exception("Test error")):
            result = run_qa(args)
            assert result == 1
            mock_print.assert_called_with("❌ Q&A session failed: Test error")


class TestInteractiveLoop:
    """Test interactive loop functionality."""

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_loop_quit(self, mock_print, mock_input):
        """Test interactive loop with quit command."""
        mock_input.return_value = "quit"
        session = ChatSession("test_repo")
        args = Mock(verbose=False)

        _run_interactive_loop("test_repo", session, args)
        mock_input.assert_called_once()

    @patch("builtins.input")
    @patch("app.cli._process_question")
    @patch("builtins.print")
    def test_run_interactive_loop_normal_question(
        self, mock_print, mock_process, mock_input
    ):
        """Test interactive loop with normal question."""
        mock_input.side_effect = ["What is this?", "quit"]
        session = ChatSession("test_repo")
        args = Mock(verbose=False)

        _run_interactive_loop("test_repo", session, args)
        mock_process.assert_called_once_with("What is this?", "test_repo", session)

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_loop_empty_input(self, mock_print, mock_input):
        """Test interactive loop with empty input."""
        mock_input.side_effect = ["", "quit"]
        session = ChatSession("test_repo")
        args = Mock(verbose=False)

        _run_interactive_loop("test_repo", session, args)
        # Should not process empty input

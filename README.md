# README-Mentor

⚡️ **README-Mentor** – Chat-style, source-cited Q&A over any repo's README & docs (LangChain + Chroma + LLM)

## Overview

README-Mentor is an AI-powered tool that ingests GitHub repositories and provides intelligent Q&A capabilities over their documentation. It uses advanced text chunking, embedding generation, and vector search to enable contextual conversations about any repository's README and documentation.

## Features

- **Repository Ingestion**: Automatically fetch and process GitHub repositories
- **Smart Text Chunking**: Intelligent document splitting with line mapping
- **Vector Search**: Semantic search using sentence transformers
- **CLI Interface**: Easy-to-use command-line tools
- **Persistent Storage**: Optional ChromaDB persistence for production use
- **Comprehensive Testing**: Full test suite with edge case coverage

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- GitHub token (for private repositories)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/readme-mentor.git
   cd readme-mentor
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your GitHub token and other settings
   ```

### Basic Usage

#### CLI Ingestion

Ingest a repository using the command-line interface:

```bash
# Basic ingestion
python -m app.embeddings https://github.com/octocat/Hello-World

# Using the installed CLI command
readme-mentor-ingest https://github.com/octocat/Hello-World

# With custom parameters
python -m app.embeddings https://github.com/user/repo \
  --chunk-size 512 \
  --file-glob "*.md" "docs/**/*.md" \
  --persist-dir ./data/chroma
```

#### Programmatic Usage

```python
from app.embeddings.ingest import ingest_repository

# Ingest a repository
vectorstore = ingest_repository(
    repo_url="https://github.com/octocat/Hello-World",
    chunk_size=1024,
    chunk_overlap=128,
    file_glob=("README*", "docs/**/*.md")
)

# Search the ingested content
results = vectorstore.similarity_search("How do I get started?", k=3)
for result in results:
    print(f"File: {result.metadata['file']}")
    print(f"Lines: {result.metadata['start_line']}-{result.metadata['end_line']}")
    print(f"Content: {result.page_content[:200]}...")
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# GitHub API token (required for private repos or rate limiting)
GITHUB_TOKEN=your_github_token_here

# Optional: Custom embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: ChromaDB persistence directory
CHROMA_PERSIST_DIR=./data/chroma
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `repo_url` | GitHub repository URL to ingest | Required |
| `--file-glob` | File patterns to process | `README*`, `docs/**/*.md` |
| `--chunk-size` | Text chunk size in characters | `1024` |
| `--chunk-overlap` | Overlap between chunks | `128` |
| `--batch-size` | Embedding batch size | `64` |
| `--embedding-model` | Sentence-transformers model | `all-MiniLM-L6-v2` |
| `--collection-name` | Custom ChromaDB collection name | Auto-generated |
| `--persist-dir` | ChromaDB persistence directory | In-memory |
| `--verbose` | Enable verbose logging | False |

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_ingest_test.py -v
```

### Code Quality

```bash
# Lint code
poetry run ruff check .

# Format code
poetry run ruff format .

# Run pre-commit hooks
poetry run pre-commit run --all-files
```

### Project Structure

```
readme-mentor/
├── app/
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── __main__.py      # CLI entry point
│   │   └── ingest.py        # Core ingestion logic
│   ├── github/
│   │   ├── __init__.py
│   │   └── loader.py        # GitHub file fetching
│   ├── preprocess/
│   │   ├── __init__.py
│   │   └── markdown_cleaner.py  # Text cleaning
│   └── utils/
│       ├── __init__.py
│       └── validators.py    # Input validation
├── tests/                   # Test suite
├── data/                    # Downloaded files and ChromaDB
└── docs/                    # Documentation
```

## Testing

The project includes comprehensive tests covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Edge Cases**: Error handling and boundary conditions
- **CLI Tests**: Command-line interface validation

### Test Coverage

Current test coverage targets:
- **Minimum**: 85% overall coverage
- **Critical Paths**: 100% coverage for ingestion pipeline
- **Edge Cases**: Comprehensive error handling tests

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run the test suite: `poetry run pytest`
6. Commit with conventional commits: `git commit -m "feat: add new feature"`
7. Push and create a pull request

### Coding Standards

- **PEP 8 + Ruff**: Auto-formatting and linting
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Comprehensive documentation
- **Conventional Commits**: Structured commit messages
- **Test Coverage**: Maintain high coverage standards

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version History

- **v0.2.0**: Complete ingestion pipeline with CLI interface
- **v0.1.0**: Initial project setup and basic structure

## Support

For questions, issues, or contributions, please:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/your-username/readme-mentor/issues)
3. Create a new issue with detailed information
4. Join our community discussions

---

**Built with ❤️ using LangChain, ChromaDB, and Sentence Transformers**

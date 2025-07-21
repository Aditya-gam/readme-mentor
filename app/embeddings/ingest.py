"""Text chunking and embedding ingestion pipeline for readme-mentor.

This module provides the core ingestion routine that processes repository content
through text chunking, embedding generation, and vector store storage.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..github.loader import fetch_repository_files
from ..preprocess.markdown_cleaner import LineOffsetMapper, clean_markdown_file
from ..utils.validators import InvalidRepoURLError, validate_repo_url

# Configure logging
logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_BATCH_SIZE = 64

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _extract_repo_slug(repo_url: str) -> str:
    """Extract repository slug from GitHub URL.

    Args:
        repo_url: GitHub repository URL

    Returns:
        Repository slug (e.g., 'octocat_Hello-World')
    """
    # Clean the URL by removing query parameters, fragments, and extra paths
    cleaned_url = repo_url.split("?")[0].split("#")[0].rstrip("/")

    # Extract owner and repo name from URL
    if "github.com" in cleaned_url:
        parts = cleaned_url.split("/")
        # Find the github.com part and get the next two parts
        try:
            github_index = parts.index("github.com")
            if len(parts) > github_index + 2:
                owner = parts[github_index + 1]
                repo_name = parts[github_index + 2].replace(".git", "")
                return f"{owner}_{repo_name}"
        except ValueError:
            pass

    # Fallback: use URL as slug
    return cleaned_url.replace("https://", "").replace("http://", "").replace("/", "_")


def _create_persist_directory(repo_slug: str) -> Path:
    """Create and return the persistence directory for vector store.

    Args:
        repo_slug: Repository slug

    Returns:
        Path to the persistence directory
    """
    persist_dir = Path("data") / repo_slug / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


def _generate_collection_name(repo_slug: str) -> str:
    """Generate a unique collection name for ChromaDB.

    Args:
        repo_slug: Repository slug

    Returns:
        Unique collection name
    """
    unique_id = str(uuid.uuid4())[:8]
    return f"{repo_slug}_{unique_id}"


def _chunk_text_with_line_mapping(
    text: str,
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Chunk text and preserve line mapping information.

    Args:
        text: Text content to chunk
        file_path: Path to the source file
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of Document objects with metadata
    """
    # Create line mapper for the original text
    line_mapper = LineOffsetMapper(text)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Convert chunks to Document objects with metadata
    documents = []
    current_pos = 0

    for chunk in chunks:
        # Find the character range for this chunk in the original text
        chunk_start = text.find(chunk, current_pos)
        if chunk_start == -1:
            # Fallback: use current position and try to find the chunk
            # This can happen with overlap or when the splitter modifies the text
            chunk_start = current_pos
            # Try to find a substring that matches as much as possible
            for i in range(len(text) - len(chunk) + 1):
                if text[i : i + len(chunk)] == chunk:
                    chunk_start = i
                    break

        chunk_end = min(chunk_start + len(chunk), len(text))
        current_pos = chunk_end

        # Ensure we don't exceed text bounds
        if chunk_start >= len(text):
            chunk_start = len(text) - 1
        if chunk_end > len(text):
            chunk_end = len(text)

        # Get line range for this chunk
        try:
            start_line, end_line = line_mapper.get_line_range(chunk_start, chunk_end)
        except ValueError:
            # Fallback: use the entire text range
            start_line, end_line = 0, len(text.split("\n")) - 1

        # Create metadata
        metadata = {
            "file": str(file_path),
            "start_line": start_line,
            "end_line": end_line,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        }

        # Create Document object
        document = Document(page_content=chunk, metadata=metadata)
        documents.append(document)

    return documents


def _process_file_for_chunking(file_path: Path) -> List[Document]:
    """Process a single file for chunking.

    Args:
        file_path: Path to the file to process

    Returns:
        List of Document objects with chunks
    """
    try:
        logger.info(f"Processing file for chunking: {file_path}")

        # Clean the markdown content
        cleaned_doc = clean_markdown_file(file_path, include_code=False)

        # Chunk the cleaned text
        documents = _chunk_text_with_line_mapping(cleaned_doc.text, file_path)

        logger.info(f"Created {len(documents)} chunks from {file_path}")
        return documents

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        return []


def _generate_embeddings_batch(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[List[float]]:
    """Generate embeddings for documents in batches.

    Args:
        documents: List of Document objects
        embedding_model: HuggingFace embedding model
        batch_size: Number of documents to process in each batch

    Returns:
        List of embedding vectors
    """
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_texts = [doc.page_content for doc in batch]

        logger.info(
            f"Generating embeddings for batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}"
        )

        try:
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {e}")
            # Add zero vectors as fallback
            zero_vector = [0.0] * 384  # Default dimension for MiniLM
            all_embeddings.extend([zero_vector] * len(batch))

    return all_embeddings


def ingest_repository(
    repo_url: str,
    file_glob: Optional[tuple[str, ...]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Ingest a repository for vector search.

    This function performs the complete ingestion pipeline:
    1. Validates the repository URL
    2. Fetches repository files
    3. Cleans and chunks the content
    4. Generates embeddings
    5. Stores in ChromaDB vector store

    Args:
        repo_url: GitHub repository URL
        file_glob: Glob patterns for files to process (default: README* and docs/**/*.md)
        chunk_size: Size of each text chunk in characters
        chunk_overlap: Overlap between chunks in characters
        batch_size: Number of documents to process in embedding batches
        embedding_model_name: Name of the sentence-transformers model to use
        collection_name: Custom collection name for ChromaDB
        persist_directory: Directory to persist ChromaDB data

    Returns:
        ChromaDB vector store instance

    Raises:
        InvalidRepoURLError: If the repository URL is invalid
        Exception: If ingestion fails
    """
    logger.info(f"Starting ingestion for repository: {repo_url}")

    # Validate repository URL
    try:
        validated_url = validate_repo_url(repo_url)
        logger.info(f"Repository URL validated: {validated_url}")
    except (ValueError, InvalidRepoURLError) as e:
        logger.error(f"Invalid repository URL: {e}")
        raise

    # Extract repository slug
    repo_slug = _extract_repo_slug(validated_url)
    logger.info(f"Repository slug: {repo_slug}")

    # Set default file patterns if not provided
    if file_glob is None:
        file_glob = ("README*", "docs/**/*.md")

    # Fetch repository files
    logger.info("Fetching repository files...")
    file_paths = fetch_repository_files(validated_url, file_glob)

    if not file_paths:
        logger.warning("No files found in repository")
        raise Exception("No files found in repository")

    logger.info(f"Found {len(file_paths)} files to process")

    # Process files for chunking
    all_documents = []
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        if file_path.exists():
            documents = _process_file_for_chunking(file_path)
            all_documents.extend(documents)

    if not all_documents:
        logger.warning("No documents created from files")
        raise Exception("No documents created from files")

    logger.info(f"Created {len(all_documents)} total chunks")

    # Initialize embedding model
    logger.info(f"Initializing embedding model: {embedding_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = _generate_embeddings_batch(all_documents, embedding_model, batch_size)

    if len(embeddings) != len(all_documents):
        logger.error(
            f"Embedding count mismatch: {len(embeddings)} vs {len(all_documents)}"
        )
        raise Exception("Embedding generation failed")

    logger.info(f"Generated embeddings for {len(embeddings)} chunks")

    # Create vector store
    logger.info("Creating vector store...")
    if collection_name is None:
        collection_name = _generate_collection_name(repo_slug)

    # Create ChromaDB vector store
    if persist_directory is None:
        # Use in-memory storage
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
    else:
        # Use persistent storage
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory,
        )

    # Add documents to the vector store
    texts = [doc.page_content for doc in all_documents]
    metadatas = [doc.metadata for doc in all_documents]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    # ChromaDB automatically persists in newer versions
    # No need to call persist() manually

    logger.info(f"ChromaDB index built at {persist_directory}")
    logger.info(f"Collection name: {collection_name}")
    logger.info(f"Total documents stored: {len(all_documents)}")

    return vectorstore

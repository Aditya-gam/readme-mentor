"""Text chunking and embedding ingestion pipeline for readme-mentor.

This module provides the core ingestion routine that processes repository content
through text chunking, embedding generation, and vector store storage.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..github.loader import fetch_repository_files
from ..preprocess.markdown_cleaner import LineOffsetMapper, clean_markdown_file
from ..utils.validators import validate_repo_url

# Configure logging
logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_BATCH_SIZE = 64

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Get a HuggingFace embedding model instance.

    Args:
        model_name: Name of the sentence-transformers model to use

    Returns:
        HuggingFaceEmbeddings instance

    Raises:
        Exception: If the embedding model cannot be loaded
    """
    try:
        logger.info(f"Initializing embedding model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise RuntimeError(f"Failed to load embedding model: {e}") from e


def get_vector_store(
    repo_id: str, embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """
    Load or create a Chroma vector store for the given repository ID.

    Args:
        repo_id: Repository identifier (e.g., 'owner_repo')
        embedding_model: HuggingFace embedding model instance (optional)

    Returns:
        Chroma vector store instance

    Raises:
        ValueError: If the vector store cannot be found or loaded
        Exception: If there are issues with the embedding model
    """
    # Construct the path to the vector store
    vector_store_path = Path("data") / repo_id / "chroma"

    if not vector_store_path.exists():
        raise ValueError(f"Vector store not found for repository ID: {repo_id}")

    # Get embedding model if not provided
    if embedding_model is None:
        embedding_model = get_embedding_model()

    try:
        logger.info(f"Loading vector store from: {vector_store_path}")

        # Find the collection that contains documents for this repository
        import chromadb

        client = chromadb.PersistentClient(path=str(vector_store_path))
        collections = client.list_collections()

        # Look for a collection that starts with the repo_id and has documents
        target_collection = None
        for collection in collections:
            if collection.name.startswith(repo_id) and collection.count() > 0:
                target_collection = collection.name
                logger.info(f"Found collection with documents: {target_collection}")
                break

        if target_collection is None:
            raise ValueError(
                f"No collection with documents found for repository ID: {repo_id}"
            )

        # Load the vector store with the found collection
        vectorstore = Chroma(
            collection_name=target_collection,
            embedding_function=embedding_model,
            persist_directory=str(vector_store_path),
        )

        logger.info(
            f"Successfully loaded vector store with {vectorstore._collection.count()} documents"
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise ValueError(
            f"Failed to load vector store for repository ID {repo_id}: {e}"
        ) from e


def _extract_repo_slug(repo_url: str) -> str:
    """
    Extract repository slug from GitHub URL.

    Args:
        repo_url: GitHub repository URL

    Returns:
        Repository slug (e.g., 'owner_repo')

    Raises:
        ValueError: If the URL format is invalid
    """
    try:
        # Extract owner and repo from URL
        parts = repo_url.rstrip("/").split("/")
        if len(parts) >= 2:
            owner = parts[-2]
            repo = parts[-1]
            return f"{owner}_{repo}"

        raise ValueError(f"Could not extract repository slug from URL: {repo_url}")
    except Exception as e:
        raise ValueError(
            f"Failed to extract repository slug from URL '{repo_url}': {e}"
        ) from e


def _create_persist_directory(repo_slug: str) -> Path:
    """
    Create persistence directory for the repository.

    Args:
        repo_slug: Repository slug

    Returns:
        Path to the persistence directory
    """
    persist_dir = Path("data") / repo_slug / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


def _generate_collection_name(repo_slug: str) -> str:
    """
    Generate a unique collection name for the repository.

    Args:
        repo_slug: Repository slug

    Returns:
        Collection name
    """
    # Use repo_slug as base, add timestamp for uniqueness
    timestamp = int(time.time())
    return f"{repo_slug}_{timestamp}"


def _find_chunk_position(text: str, chunk: str, current_pos: int) -> int:
    """
    Find the position of a chunk within the original text.

    Args:
        text: Original text
        chunk: Text chunk to find
        current_pos: Current position to start searching from

    Returns:
        Position of the chunk in the original text
    """
    # Find the chunk in the text starting from current_pos
    pos = text.find(chunk, current_pos)
    if pos == -1:
        # If not found, try searching from the beginning
        pos = text.find(chunk)
    return pos if pos != -1 else current_pos


def _get_chunk_line_range(
    line_mapper: LineOffsetMapper, chunk_start: int, chunk_end: int, text: str
) -> tuple[int, int]:
    """
    Get the line range for a chunk using the line offset mapper.

    Args:
        line_mapper: Line offset mapper
        chunk_start: Start position of the chunk
        chunk_end: End position of the chunk
        text: Original text

    Returns:
        Tuple of (start_line, end_line)
    """
    try:
        start_line = line_mapper.get_line_number(chunk_start)
        end_line = line_mapper.get_line_number(chunk_end)
        return start_line, end_line
    except Exception:
        # Fallback: estimate line numbers
        lines_before_start = text[:chunk_start].count("\n")
        lines_in_chunk = text[chunk_start:chunk_end].count("\n")
        start_line = lines_before_start + 1
        end_line = start_line + lines_in_chunk
        return start_line, end_line


def _chunk_text_with_line_mapping(
    text: str,
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Chunk text and preserve line number information.

    Args:
        text: Text to chunk
        file_path: Path to the source file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects with metadata
    """
    # Create line offset mapper
    line_mapper = LineOffsetMapper(text)

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Create documents with metadata
    documents = []
    current_pos = 0

    for i, chunk in enumerate(chunks):
        # Find chunk position in original text
        chunk_start = _find_chunk_position(text, chunk, current_pos)
        chunk_end = chunk_start + len(chunk)
        current_pos = chunk_end

        # Get line range
        start_line, end_line = _get_chunk_line_range(
            line_mapper, chunk_start, chunk_end, text
        )

        # Create metadata
        metadata = {
            "source": str(file_path),
            "start_line": start_line,
            "end_line": end_line,
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": i,
            "file_type": file_path.suffix.lower(),
        }

        # Create document
        document = Document(page_content=chunk, metadata=metadata)
        documents.append(document)

    return documents


def _process_file_for_chunking(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Process a single file for chunking.

    Args:
        file_path: Path to the file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects
    """
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean markdown if it's a markdown file
        if file_path.suffix.lower() in [".md", ".markdown"]:
            content = clean_markdown_file(content)

        # Chunk the content
        documents = _chunk_text_with_line_mapping(
            content, file_path, chunk_size, chunk_overlap
        )

        logger.debug(f"Processed {file_path}: {len(documents)} chunks")
        return documents

    except Exception as e:
        logger.warning(f"Failed to process file {file_path}: {e}")
        return []


def _generate_embeddings_batch(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[List[float]]:
    """
    Generate embeddings for documents in batches.

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
        texts = [doc.page_content for doc in batch]

        try:
            batch_embeddings = embedding_model.embed_documents(texts)
            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Generated embeddings for batch {i // batch_size + 1}")

        except Exception as e:
            logger.error(
                f"Failed to generate embeddings for batch {i // batch_size + 1}: {e}"
            )
            # Add zero vectors for failed embeddings to maintain alignment
            for _ in batch:
                all_embeddings.append(
                    [0.0] * embedding_model.client.get_sentence_embedding_dimension()
                )

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
    user_output=None,
) -> Chroma:
    """
    Ingest a GitHub repository for Q&A.

    This function processes repository content through text chunking,
    embedding generation, and vector store storage.

    Args:
        repo_url: GitHub repository URL
        file_glob: File patterns to process (default: README* and docs/**/*.md)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        batch_size: Batch size for embedding generation
        embedding_model_name: Name of the embedding model to use
        collection_name: Optional collection name for the vector store
        persist_directory: Optional directory to persist the vector store
        user_output: UserOutput instance for progress tracking

    Returns:
        Chroma vector store instance

    Raises:
        ValueError: If the repository URL is invalid
        Exception: If ingestion fails
    """
    # Start timing the entire operation
    start_time = time.time()

    if user_output:
        user_output.start_operation_timer("ingestion")
        user_output.step("Starting repository ingestion", "ingestion")
        user_output.config_detail("repo_url", repo_url)
        user_output.config_detail("chunk_size", chunk_size)
        user_output.config_detail("chunk_overlap", chunk_overlap)
        user_output.config_detail("embedding_model", embedding_model_name)

    try:
        # Step 1: Validate and prepare repository
        if user_output:
            user_output.step(
                "Validating repository URL and extracting metadata", "ingestion"
            )

        repo_slug, repo_name = _validate_and_prepare_repo(repo_url)

        if user_output:
            user_output.detail(f"Repository slug: {repo_slug}")
            user_output.detail(f"Repository name: {repo_name}")

        # Step 2: Set up file patterns
        if file_glob is None:
            file_glob = ("README*", "docs/**/*.md")

        if user_output:
            user_output.step(f"Using file patterns: {file_glob}", "ingestion")

        # Step 3: Fetch repository files
        if user_output:
            user_output.step("Fetching repository files from GitHub", "ingestion")

        file_paths = _fetch_repository_files(repo_url, file_glob, user_output)

        if user_output:
            user_output.detail(f"Found {len(file_paths)} files to process")
            if user_output.config.show_raw_data:
                user_output.raw(file_paths, "File Paths")

        # Step 4: Process files for chunking
        if user_output:
            user_output.step("Processing files and creating text chunks", "ingestion")

        all_documents, total_chunks = _process_files_for_chunking(
            file_paths, chunk_size, chunk_overlap, user_output
        )

        if user_output:
            user_output.add_performance_metric("total_files", len(file_paths))
            user_output.add_performance_metric("total_chunks", total_chunks)
            user_output.detail(
                f"Created {total_chunks} chunks from {len(file_paths)} files"
            )

        # Step 5: Initialize embedding model
        if user_output:
            user_output.step(
                f"Initializing embedding model: {embedding_model_name}", "ingestion"
            )

        embedding_model = _initialize_embedding_model(embedding_model_name, user_output)

        if user_output:
            user_output.config_detail("embedding_model_loaded", True)

        # Step 6: Generate embeddings
        if user_output:
            user_output.step("Generating embeddings for text chunks", "ingestion")

        embeddings = _generate_embeddings_with_progress(
            all_documents, embedding_model, batch_size, user_output
        )

        if user_output:
            user_output.detail(f"Generated {len(embeddings)} embeddings")
            if user_output.config.show_raw_data:
                user_output.raw(
                    f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}",
                    "Embedding Info",
                )

        # Step 7: Create vector store
        if user_output:
            user_output.step("Creating and populating vector store", "ingestion")

        vectorstore = _create_vector_store(
            all_documents,
            embedding_model,
            repo_slug,
            collection_name,
            persist_directory,
            user_output,
        )

        if user_output:
            user_output.detail(
                f"Vector store created with collection: {vectorstore._collection.name}"
            )

        # Step 8: Track completion metrics
        duration = time.time() - start_time

        if user_output:
            user_output.end_operation_timer("ingestion")
            _log_completion_metrics(
                user_output,
                duration,
                embedding_model_name,
                vectorstore._collection.name,
                persist_directory,
                len(all_documents),
            )

            # Log internal state if in debug mode
            if user_output.config.show_internal_state:
                internal_state = {
                    "vectorstore_collection": vectorstore._collection.name,
                    "total_documents": len(all_documents),
                    "total_chunks": total_chunks,
                    "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
                    "persist_directory": persist_directory,
                }
                user_output.internal_state(internal_state)

        return vectorstore

    except Exception as e:
        if user_output:
            user_output.error(
                "Repository ingestion failed", error=e, context="ingest_repository"
            )
        logger.error(f"Ingestion failed for {repo_url}: {e}")
        raise


def _validate_and_prepare_repo(repo_url: str) -> tuple[str, str]:
    """Validate repository URL and extract slug."""
    try:
        validated_url = validate_repo_url(repo_url)
        logger.info(f"Repository URL validated: {validated_url}")
    except ValueError as e:
        logger.error(f"Invalid repository URL: {e}")
        raise

    repo_slug = _extract_repo_slug(validated_url)
    logger.info(f"Repository slug: {repo_slug}")
    return validated_url, repo_slug


def _fetch_repository_files(
    repo_url: str, file_glob: tuple[str, ...], user_output
) -> List[str]:
    """Fetch repository files from GitHub."""
    if user_output:
        user_output.step(f"Fetching files matching patterns: {file_glob}", "ingestion")

    file_paths = fetch_repository_files(repo_url, file_glob)

    if user_output:
        user_output.detail(f"Retrieved {len(file_paths)} files from GitHub")
        if user_output.config.show_raw_data:
            # Show first 10 to avoid spam
            user_output.raw(file_paths[:10], "First 10 File Paths")

    return file_paths


def _process_files_for_chunking(
    file_paths: List[str], chunk_size: int, chunk_overlap: int, user_output
) -> tuple[List, int]:
    """Process files for chunking with progress tracking."""
    if user_output and user_output.config.show_detailed_progress:
        return _process_files_with_progress(
            file_paths, chunk_size, chunk_overlap, user_output
        )
    else:
        return _process_files_simple(file_paths, chunk_size, chunk_overlap)


def _process_files_with_progress(
    file_paths: List[str], chunk_size: int, chunk_overlap: int, user_output
) -> tuple[List, int]:
    """Process files with detailed progress tracking."""
    all_documents = []
    total_chunks = 0

    with user_output.progress_bar(len(file_paths), "Processing files") as progress:
        task = progress.add_task("Processing files", total=len(file_paths))

        for i, file_path in enumerate(file_paths):
            if user_output:
                user_output.step(
                    f"Processing file {i + 1}/{len(file_paths)}: {file_path}",
                    "ingestion",
                )

            try:
                documents = _process_file_for_chunking(
                    file_path, chunk_size, chunk_overlap
                )
                all_documents.extend(documents)
                total_chunks += len(documents)

                if user_output:
                    user_output.detail(
                        f"Created {len(documents)} chunks from {file_path}"
                    )

                progress.update(task, advance=1)

            except Exception as e:
                if user_output:
                    user_output.warning(f"Failed to process {file_path}: {e}")
                logger.warning(f"Failed to process {file_path}: {e}")
                progress.update(task, advance=1)

    return all_documents, total_chunks


def _process_files_simple(
    file_paths: List[str], chunk_size: int, chunk_overlap: int
) -> tuple[List, int]:
    """Process files without detailed progress tracking."""
    all_documents = []
    total_chunks = 0

    for file_path in file_paths:
        try:
            documents = _process_file_for_chunking(file_path, chunk_size, chunk_overlap)
            all_documents.extend(documents)
            total_chunks += len(documents)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")

    return all_documents, total_chunks


def _track_processing_metrics(
    user_output,
    processed_files: int,
    total_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
):
    """Track processing metrics for user output."""
    if user_output:
        user_output.add_performance_metric("processed_files", processed_files)
        user_output.add_performance_metric("total_chunks", total_chunks)
        user_output.add_performance_metric("chunk_size", chunk_size)
        user_output.add_performance_metric("chunk_overlap", chunk_overlap)

        if user_output.config.show_detailed_metrics:
            avg_chunks_per_file = (
                total_chunks / processed_files if processed_files > 0 else 0
            )
            user_output.add_performance_metric(
                "avg_chunks_per_file", avg_chunks_per_file
            )


def _initialize_embedding_model(
    embedding_model_name: str, user_output
) -> HuggingFaceEmbeddings:
    """Initialize the embedding model."""
    if user_output:
        user_output.step(
            f"Loading embedding model: {embedding_model_name}", "ingestion"
        )

    embedding_model = get_embedding_model(embedding_model_name)

    if user_output:
        user_output.detail("Embedding model loaded successfully")
        user_output.config_detail("embedding_model_name", embedding_model_name)

    return embedding_model


def _generate_embeddings_with_progress(
    all_documents: List,
    embedding_model: HuggingFaceEmbeddings,
    batch_size: int,
    user_output,
) -> List[List[float]]:
    """Generate embeddings with progress tracking."""
    if user_output:
        user_output.step(
            f"Generating embeddings for {len(all_documents)} documents", "ingestion"
        )
        user_output.config_detail("batch_size", batch_size)

    embeddings = _generate_embeddings_batch(all_documents, embedding_model, batch_size)

    if user_output:
        user_output.detail(f"Generated {len(embeddings)} embeddings")
        if user_output.config.show_detailed_metrics:
            user_output.add_performance_metric(
                "embedding_dimensions", len(embeddings[0]) if embeddings else 0
            )
            user_output.add_performance_metric("total_embeddings", len(embeddings))

    return embeddings


def _create_vector_store(
    all_documents: List,
    embedding_model: HuggingFaceEmbeddings,
    repo_slug: str,
    collection_name: Optional[str],
    persist_directory: Optional[str],
    user_output,
) -> Chroma:
    """Create and populate the vector store."""
    if user_output:
        user_output.step("Creating vector store and adding documents", "ingestion")
        user_output.config_detail("collection_name", collection_name)
        user_output.config_detail("persist_directory", persist_directory)

    vectorstore = _create_chroma_store(
        all_documents, embedding_model, repo_slug, collection_name, persist_directory
    )

    if user_output:
        user_output.detail(f"Vector store created with {len(all_documents)} documents")
        user_output.config_detail(
            "vectorstore_collection", vectorstore._collection.name
        )

    return vectorstore


def _create_chroma_store(
    all_documents: List,
    embedding_model: HuggingFaceEmbeddings,
    repo_slug: str,
    collection_name: Optional[str],
    persist_directory: Optional[str],
) -> Chroma:
    """Create ChromaDB vector store."""
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
    return vectorstore


def _log_completion_metrics(
    user_output,
    duration: float,
    embedding_model_name: str,
    collection_name: str,
    persist_directory: Optional[str],
    total_documents: int,
):
    """Log completion metrics."""
    if user_output:
        user_output.add_performance_metric("total_duration", duration)
        user_output.add_performance_metric("embedding_model_used", embedding_model_name)
        user_output.add_performance_metric("collection_name", collection_name)
        user_output.add_performance_metric("total_documents", total_documents)

        if persist_directory:
            user_output.add_performance_metric("persist_directory", persist_directory)

        if user_output.config.show_detailed_metrics:
            docs_per_second = total_documents / duration if duration > 0 else 0
            user_output.add_performance_metric("docs_per_second", docs_per_second)

"""
Backend QA Interface Module

This module provides the core backend logic for the readme-mentor QA system.
It ties together the vector store, LLM, and citation components to generate
answers to user queries about repository documentation.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from pydantic import ValidationError

from app.embeddings.ingest import get_embedding_model, get_vector_store
from app.llm.provider import get_chat_model
from app.models import QuestionPayload
from app.rag.chain import get_qa_chain
from app.rag.citations import render_citations

# Configure logging
logger = logging.getLogger(__name__)


def generate_answer(
    query: str, repo_id: str, history: List[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate an answer to a given query using a RAG-based QA chain.

    This function forms the core backend logic tying everything together:
    1. Validates input using Pydantic models
    2. Loads the vector store for the given repository
    3. Gets the LLM and QA chain
    4. Executes the chain with conversation history
    5. Post-processes the answer with citations
    6. Measures and reports latency
    7. Returns structured output with answer, citations, and metadata

    Args:
        query: The user's question string (non-empty)
        repo_id: The ID of the repository (e.g., 'owner_repo')
        history: A list of prior QA pairs for conversation context (default: empty list)

    Returns:
        A dictionary containing:
        - "answer": The final answer string with markdown citations
        - "citations": List of citation metadata dictionaries
        - "latency_ms": Response time in milliseconds

    Raises:
        ValueError: If input validation fails or vector store not found
        RuntimeError: If LLM backend is not available
        Exception: For other processing errors
    """
    # Initialize history if None
    if history is None:
        history = []

    start_time = time.perf_counter()

    try:
        # Step 1: Validate input payload using Pydantic model
        logger.info(f"Validating input for repo_id: {repo_id}")
        payload = QuestionPayload(query=query, repo_id=repo_id, history=history)

        # Extract validated values
        repo_id = payload.repo_id
        query = payload.query
        history = payload.history

        logger.info(f"Input validation passed. Query: '{query[:50]}...'")

        # Step 2: Load or retrieve the vector store for the given repo_id
        logger.info(f"Loading vector store for repository: {repo_id}")
        embedding_model = get_embedding_model()
        vector_store = get_vector_store(repo_id, embedding_model)

        # Step 3: Get the LLM and QA chain
        logger.info("Initializing LLM and QA chain")
        llm = get_chat_model()
        qa_chain = get_qa_chain(vector_store, llm)

        # Step 4: Prepare conversation input
        # Convert history to the format expected by ConversationalRetrievalChain
        # The chain expects a list of [human, AI] message tuples
        chat_history = []
        for human_msg, ai_msg in history:
            chat_history.append((human_msg, ai_msg))

        logger.info(
            f"Prepared chat history with {len(chat_history)} previous exchanges"
        )

        # Step 5: Execute the chain with return_source_documents=True
        logger.info("Executing QA chain")
        result = qa_chain({"question": query, "chat_history": chat_history})

        # Extract the raw answer and source documents
        raw_answer = result.get("answer", "")
        source_documents = result.get("source_documents", [])

        if not raw_answer:
            logger.warning("QA chain returned empty answer")
            raw_answer = "I couldn't generate an answer for your question."

        logger.info(
            f"QA chain execution completed. Retrieved {len(source_documents)} source documents"
        )

        # Step 6: Post-process the LLM's answer with citations
        logger.info("Processing answer with citations")
        final_answer = render_citations(raw_answer, source_documents)

        # Step 7: Prepare structured citation metadata
        citations = _extract_citation_metadata(source_documents)

        # Step 8: Measure latency
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        logger.info(f"Answer generation completed in {latency_ms}ms")

        # Step 9: Return structured output
        return {
            "answer": final_answer,
            "citations": citations,
            "latency_ms": latency_ms,
        }

    except ValidationError as e:
        logger.error(f"Input validation failed: {e.errors()}")
        raise ValueError(f"Invalid input payload: {e.errors()}") from e

    except ValueError as e:
        logger.error(f"Value error during processing: {e}")
        raise

    except RuntimeError as e:
        logger.error(f"Runtime error (likely LLM backend issue): {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during answer generation: {e}")
        raise Exception(f"Failed to generate answer: {e}") from e


def _extract_citation_metadata(
    source_documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Extract structured citation metadata from source documents.

    Args:
        source_documents: List of Document objects used in the answer

    Returns:
        List of citation metadata dictionaries
    """
    citations = []
    seen_files = set()  # Track unique files to avoid duplicates

    for doc in source_documents:
        metadata = doc.metadata

        # Extract file information
        file_name = metadata.get("file") or metadata.get("source")
        start_line = metadata.get("start_line") or metadata.get("line_start")
        end_line = metadata.get("end_line") or metadata.get("line_end")

        # Validate metadata fields
        if not all([file_name, start_line is not None, end_line is not None]):
            logger.warning(f"Skipping document with incomplete metadata: {metadata}")
            continue

        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except (ValueError, TypeError):
            logger.warning(f"Skipping document with invalid line numbers: {metadata}")
            continue

        # Create citation entry
        citation = {
            "file": str(file_name),
            "start_line": start_line,
            "end_line": end_line,
        }

        # Only add if we haven't seen this file before (avoid duplicates)
        file_key = f"{file_name}:{start_line}-{end_line}"
        if file_key not in seen_files:
            citations.append(citation)
            seen_files.add(file_key)

    logger.info(
        f"Extracted {len(citations)} unique citations from {len(source_documents)} source documents"
    )
    return citations

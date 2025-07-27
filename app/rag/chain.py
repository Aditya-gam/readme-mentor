"""
ConversationalRetrievalChain Assembly Module

This module implements the retrieval-augmented QA chain using LangChain's
ConversationalRetrievalChain with custom document formatting and memory management.
"""

import os
import time
import warnings
from typing import Any, List

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.metrics import get_metrics_collector
from app.metrics.models import ErrorCategory, ToolCallStatus
from app.rag.citations import render_citations

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*ConversationBufferWindowMemory.*")
warnings.filterwarnings("ignore", message=".*LLMChain.*")
warnings.filterwarnings("ignore", message=".*Please see the migration guide.*")
warnings.filterwarnings("ignore", message=".*migrating_memory.*")


class HybridREADMERetriever(BaseRetriever):
    """Hybrid retriever that combines similarity search with keyword matching for better README prioritization."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, vector_store: VectorStore, k: int = 4):
        """Initialize the retriever.

        Args:
            vector_store: The underlying vector store
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.k = k
        self.base_retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k * 3}
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid approach.

        Args:
            query: The search query

        Returns:
            List of relevant documents with README files prioritized
        """
        try:
            # First, try to get README-specific results
            readme_docs = self._get_readme_documents(query)

            # If we have enough README docs, return them
            if len(readme_docs) >= self.k:
                return readme_docs[: self.k]

            # Otherwise, get general results and prioritize README
            general_docs = self.base_retriever.get_relevant_documents(query)

            # Separate README from other docs
            readme_from_general = []
            other_docs = []

            for doc in general_docs:
                if self._is_readme_document(doc):
                    readme_from_general.append(doc)
                else:
                    other_docs.append(doc)

            # Combine README docs (prioritizing README-specific results)
            all_readme = readme_docs + readme_from_general
            # Take up to half from README
            all_readme = all_readme[: self.k // 2]

            # Fill remaining slots with other docs
            remaining_slots = self.k - len(all_readme)
            result = all_readme + other_docs[:remaining_slots]

            return result[: self.k]
        except Exception:
            # Fallback to base retriever if there's any issue
            return self.base_retriever.get_relevant_documents(query)

    def _get_readme_documents(self, query: str) -> List[Document]:
        """Get documents specifically from README files.

        Args:
            query: The search query

        Returns:
            List of README documents
        """
        try:
            # Get more documents to filter
            all_docs = self.base_retriever.get_relevant_documents(query)

            # Filter for README documents
            readme_docs = [doc for doc in all_docs if self._is_readme_document(doc)]

            return readme_docs
        except Exception:
            return []

    def _is_readme_document(self, doc: Document) -> bool:
        """Check if a document is from a README file.

        Args:
            doc: The document to check

        Returns:
            True if the document is from a README file
        """
        source = doc.metadata.get("source", "")
        file_name = doc.metadata.get("file", "")

        # Check if this is a README file (case insensitive)
        return (
            "README" in source.upper()
            or "README" in file_name.upper()
            or source.endswith("README.md")
            or file_name.endswith("README.md")
        )


class READMEPrioritizingRetriever(BaseRetriever):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    """Custom retriever that prioritizes README files in search results."""

    def __init__(self, vector_store: VectorStore, k: int = 4):
        """Initialize the retriever.

        Args:
            vector_store: The underlying vector store
            k: Number of documents to retrieve
        """
        super().__init__()
        self.vector_store = vector_store
        self.k = k
        self.base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k * 4},  # Get more results to filter
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents, prioritizing README files.

        Args:
            query: The search query

        Returns:
            List of relevant documents with README files prioritized
        """
        # Get documents from base retriever
        docs = self.base_retriever.get_relevant_documents(query)

        # Separate README files from other documents
        readme_docs = []
        other_docs = []

        for doc in docs:
            source = doc.metadata.get("source", "")
            file_name = doc.metadata.get("file", "")

            # Check if this is a README file (case insensitive)
            is_readme = (
                "README" in source.upper()
                or "README" in file_name.upper()
                or source.endswith("README.md")
                or file_name.endswith("README.md")
            )

            if is_readme:
                readme_docs.append(doc)
            else:
                other_docs.append(doc)

        # More aggressive README prioritization: take up to 90% from README if available
        readme_count = min(len(readme_docs), max(1, int(self.k * 0.9)))
        other_count = self.k - readme_count

        result = readme_docs[:readme_count] + other_docs[:other_count]

        # If we don't have enough README files, fill with other docs
        if len(result) < self.k:
            remaining_docs = [doc for doc in docs if doc not in result]
            result.extend(remaining_docs[: self.k - len(result)])

        return result[: self.k]

    def _get_relevant_documents_with_fallback(self, query: str) -> List[Document]:
        """Get relevant documents with README fallback strategy.

        If the initial search doesn't return enough README content,
        perform a secondary search specifically for README files.

        Args:
            query: The search query

        Returns:
            List of relevant documents with README files prioritized
        """
        # First, try normal retrieval
        docs = self._get_relevant_documents(query)

        # Check if we have enough README content
        readme_count = sum(
            1 for doc in docs if "README" in doc.metadata.get("source", "").upper()
        )

        # If we have less than 50% README content and the query seems like it should be in README,
        # try a README-specific search
        if readme_count < len(docs) * 0.5 and self._should_check_readme(query):
            readme_docs = self._search_readme_specific(query)
            if readme_docs:
                # Replace some non-README docs with README docs
                non_readme_docs = [
                    doc
                    for doc in docs
                    if "README" not in doc.metadata.get("source", "").upper()
                ]
                result = (
                    readme_docs[: min(len(readme_docs), self.k // 2)]
                    + non_readme_docs[: self.k // 2]
                )
                return result[: self.k]

        return docs

    def _should_check_readme(self, query: str) -> bool:
        """Determine if a query should prioritize README files.

        Args:
            query: The search query

        Returns:
            True if the query should prioritize README files
        """
        query_lower = query.lower()
        readme_keywords = [
            "how to",
            "how do i",
            "getting started",
            "install",
            "setup",
            "usage",
            "quick start",
            "basic",
            "simple",
            "command",
            "run",
            "what is",
            "overview",
            "introduction",
            "guide",
        ]
        return any(keyword in query_lower for keyword in readme_keywords)

    def _search_readme_specific(self, query: str) -> List[Document]:
        """Perform a README-specific search.

        Args:
            query: The search query

        Returns:
            List of README documents
        """
        try:
            # Get all documents and filter for README files
            all_docs = self.base_retriever.get_relevant_documents(query)
            readme_docs = [
                doc
                for doc in all_docs
                if "README" in doc.metadata.get("source", "").upper()
            ]
            return readme_docs[: self.k]
        except Exception:
            return []


def _format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string with <doc_i> tags.

    Args:
        docs: List of Document objects to format

    Returns:
        Formatted string with each document wrapped in <doc_i> tags
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_docs.append(f"<doc_{i}>{doc.page_content}</doc_{i}>")
    return "\n".join(formatted_docs)


class CustomStuffDocumentsChain(StuffDocumentsChain):
    """
    Custom StuffDocumentsChain that formats documents with <doc_i> tags
    before passing them to the underlying LLM chain.

    This ensures that when the LLM generates an answer, it can include
    <doc_0>, <doc_1>, etc. tokens as placeholders corresponding to each source.
    """

    _source_docs: List[Document] = []  # To store source documents

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict[str, Any]:
        """
        Override the parent method to format documents with custom tags.

        Args:
            docs: List of documents to process
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing the formatted documents and other inputs
        """
        self._source_docs = docs  # Store the source documents
        # Call the super method to get basic inputs
        inputs = super()._get_inputs(docs, **kwargs)
        # Override the document variable with our custom formatted documents
        inputs[self.document_variable_name] = _format_docs(docs)
        return inputs

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Override the parent method to post-process the LLM's answer with citations.

        Args:
            inputs: Dictionary containing the inputs to the chain

        Returns:
            Dictionary containing the processed answer
        """
        # Get metrics collector for tracking
        collector = get_metrics_collector()

        # Call the parent method to get the raw answer
        start_time = time.time()
        try:
            result = super()._call(inputs)
            duration = time.time() - start_time

            # Record successful LLM inference
            if hasattr(collector, "active_operations") and collector.active_operations:
                operation_id = next(iter(collector.active_operations.keys()), None)
                if operation_id:
                    collector.add_component_timing(
                        operation_id, "llm_inference", duration
                    )
                    collector.record_tool_call(
                        operation_id=operation_id,
                        tool_name="llm_inference",
                        status=ToolCallStatus.SUCCESS,
                        duration=duration,
                        metadata={"documents_count": len(self._source_docs)},
                    )

        except Exception as e:
            duration = time.time() - start_time

            # Record failed LLM inference
            if hasattr(collector, "active_operations") and collector.active_operations:
                operation_id = next(iter(collector.active_operations.keys()), None)
                if operation_id:
                    collector.add_component_timing(
                        operation_id, "llm_inference", duration
                    )
                    collector.record_tool_call(
                        operation_id=operation_id,
                        tool_name="llm_inference",
                        status=ToolCallStatus.FAILURE,
                        duration=duration,
                        error_category=ErrorCategory.UNKNOWN,
                        error_message=str(e),
                        metadata={"documents_count": len(self._source_docs)},
                    )
            raise

        # Get the raw answer from the result
        raw_answer = result[self.output_key]

        # Post-process the answer with citations
        citation_start_time = time.time()
        try:
            processed_answer = render_citations(raw_answer, self._source_docs)
            citation_duration = time.time() - citation_start_time

            # Record citation processing
            if hasattr(collector, "active_operations") and collector.active_operations:
                operation_id = next(iter(collector.active_operations.keys()), None)
                if operation_id:
                    collector.add_component_timing(
                        operation_id, "citation_processing", citation_duration
                    )

        except Exception as e:
            citation_duration = time.time() - citation_start_time

            # Record citation processing failure
            if hasattr(collector, "active_operations") and collector.active_operations:
                operation_id = next(iter(collector.active_operations.keys()), None)
                if operation_id:
                    collector.add_component_timing(
                        operation_id, "citation_processing", citation_duration
                    )
                    collector.record_tool_call(
                        operation_id=operation_id,
                        tool_name="citation_processing",
                        status=ToolCallStatus.FAILURE,
                        duration=citation_duration,
                        error_category=ErrorCategory.VALIDATION,
                        error_message=str(e),
                    )
            # Continue without citations rather than failing completely
            processed_answer = raw_answer

        # Return the processed answer under the chain's output_key
        return {self.output_key: processed_answer}


def _load_system_prompt() -> str:
    """
    Load the system prompt from the prompts directory.

    Returns:
        The content of the system prompt file

    Raises:
        FileNotFoundError: If the system prompt file cannot be found
        IOError: If there are issues reading the file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    system_prompt_path = os.path.join(current_dir, "..", "prompts", "system.txt")

    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"System prompt file not found at {system_prompt_path}. "
            "Please ensure the prompts/system.txt file exists."
        ) from None
    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}") from e


def get_qa_chain(
    vector_store: VectorStore,
    llm: BaseChatModel,
    memory_window: int = 6,
    retrieval_k: int = 4,
) -> ConversationalRetrievalChain:
    """
    Construct a retrieval-augmented QA chain using LangChain's ConversationalRetrievalChain.

    This function assembles the components for a conversational QA system that:
    - Uses Max Marginal Relevance (MMR) retrieval for diverse document selection
    - Maintains conversation history with a limited window
    - Formats retrieved documents with <doc_i> markers for source attribution
    - Integrates a system prompt for consistent answer generation

    Args:
        vector_store: The vector store to use for document retrieval
        llm: The language model instance for generating responses
        memory_window: Number of conversation turns to keep in memory (default: 6)
        retrieval_k: Number of documents to retrieve with MMR (default: 4)

    Returns:
        A configured ConversationalRetrievalChain instance

    Raises:
        FileNotFoundError: If the system prompt file cannot be found
        IOError: If there are issues reading the system prompt file
        ValueError: If invalid parameters are provided

    Example:
        >>> from app.llm.provider import get_chat_model
        >>> from app.embeddings.ingest import get_vector_store
        >>>
        >>> llm = get_chat_model()
        >>> vector_store = get_vector_store("path/to/repository")
        >>> qa_chain = get_qa_chain(vector_store, llm)
        >>>
        >>> # Use the chain
        >>> result = qa_chain({"question": "What is the main purpose of this project?"})
        >>> print(result["answer"])
    """
    # Validate inputs
    if memory_window < 1:
        raise ValueError("memory_window must be at least 1")
    if retrieval_k < 1:
        raise ValueError("retrieval_k must be at least 1")

    # Load the system prompt
    system_prompt_content = _load_system_prompt()

    # Configure the retriever with README prioritization
    retriever = READMEPrioritizingRetriever(vector_store, k=retrieval_k)

    # Initialize conversation memory with a limited window
    # This keeps the last 'memory_window' question-answer pairs in context,
    # preventing indefinite growth of conversation history
    # Note: Using the modern memory pattern to avoid deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
        )

    # Define the prompt template for the LLM
    # This combines chat history, context (with <doc_i> markers), and the current question
    qa_llm_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt_content),
            (
                "human",
                "Context: {context}\n\nQuestion: {question}\n\nRemember: If a simple answer exists in the README, use it. Don't overcomplicate or elaborate unnecessarily. Be direct and concise.",
            ),
        ]
    )

    # Create the LLMChain that will be used by our custom combine documents chain
    # Note: Using LLMChain because StuffDocumentsChain expects LLMChain.
    # TODO: Update when LangChain provides a migration path for StuffDocumentsChain
    from langchain.chains.llm import LLMChain

    qa_llm_chain = LLMChain(llm=llm, prompt=qa_llm_prompt)

    # Create the custom combine documents chain
    # This chain formats retrieved documents with <doc_i> tags and passes them
    # as 'context' to the qa_llm_chain
    combine_docs_chain = CustomStuffDocumentsChain(
        llm_chain=qa_llm_chain,
        document_variable_name="context",  # Must match the placeholder in qa_llm_prompt
    )

    # Assemble the ConversationalRetrievalChain
    # We use the from_llm method with combine_docs_chain_kwargs to ensure proper configuration
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # LLM for question generation and final answer
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Useful for testing and debugging
        verbose=False,  # Set to True for debugging chain execution
        combine_docs_chain_kwargs={
            "prompt": qa_llm_prompt,
            "document_variable_name": "context",
        },
    )

    # Override the combine_docs_chain with our custom implementation
    # This ensures documents are formatted with <doc_i> markers
    qa_chain.combine_docs_chain = combine_docs_chain

    return qa_chain

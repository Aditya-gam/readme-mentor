"""
ConversationalRetrievalChain Assembly Module

This module implements the retrieval-augmented QA chain using LangChain's
ConversationalRetrievalChain with custom document formatting and memory management.
"""

import os
import warnings
from typing import Any, List

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from app.rag.citations import render_citations

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*ConversationBufferWindowMemory.*")
warnings.filterwarnings("ignore", message=".*LLMChain.*")


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
        # Call the parent method to get the raw answer
        result = super()._call(inputs)

        # Get the raw answer from the result
        raw_answer = result[self.output_key]

        # Post-process the answer with citations
        processed_answer = render_citations(raw_answer, self._source_docs)

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

    # Configure the retriever with Max Marginal Relevance (MMR)
    # This ensures up to 'retrieval_k' relevant chunks are retrieved with diversity,
    # helping avoid redundant similar text common in README files
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": retrieval_k}
    )

    # Initialize conversation memory with a limited window
    # This keeps the last 'memory_window' question-answer pairs in context,
    # preventing indefinite growth of conversation history
    # Note: Using the modern memory pattern to avoid deprecation warnings
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
            ("human", "Context: {context}\n\nQuestion: {question}"),
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

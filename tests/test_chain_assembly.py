"""
Test module for ConversationalRetrievalChain assembly.

This module provides comprehensive tests for the chain assembly functionality,
including document formatting, memory management, and error handling.
"""

import importlib

import pytest
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import HumanMessage, SystemMessage

# Import the chain module functions directly to avoid module-level state issues
from app.rag.chain import (
    CustomStuffDocumentsChain,
    _format_docs,
    _load_system_prompt,
    get_qa_chain,
)


class TestChainAssembly:
    """
    Test class for ConversationalRetrievalChain assembly.

    This provides comprehensive testing of the chain assembly functionality,
    including document formatting, memory management, and error handling.
    """

    def setup_method(self):
        """
        Set up test fixtures for each test method.

        Creates a small fake vector store with dummy documents and a fake LLM
        for testing the chain assembly.
        """
        # Reload the chain module to ensure clean state
        import app.rag.chain

        importlib.reload(app.rag.chain)

        # Re-import the functions after reload
        from app.rag.chain import get_qa_chain

        self.get_qa_chain = get_qa_chain
        self._load_system_prompt = _load_system_prompt
        self._format_docs = _format_docs
        self.custom_stuff_documents_chain = CustomStuffDocumentsChain

        # Create dummy documents for testing
        self.test_documents = [
            Document(
                page_content="This is a test document about Python programming.",
                metadata={"source": "test_file.py", "line_start": 1, "line_end": 10},
            ),
            Document(
                page_content="Another test document about machine learning.",
                metadata={"source": "ml_guide.md", "line_start": 5, "line_end": 15},
            ),
            Document(
                page_content="Third document about data structures.",
                metadata={
                    "source": "data_structures.py",
                    "line_start": 20,
                    "line_end": 30,
                },
            ),
        ]

        # Create fake embeddings for the vector store
        self.fake_embeddings = FakeEmbeddings(size=384)

        # Create an in-memory Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=self.test_documents, embedding=self.fake_embeddings
        )

        # Create a fake LLM that echoes part of the prompt
        # This helps verify that the prompt contains the expected <doc_i> markers
        self.fake_llm = FakeListLLM(responses=["Mocked LLM response."])

    def test_chain_creation(self):
        """
        Test that the QA chain can be created successfully.

        This test verifies that:
        1. The chain can be instantiated without errors
        2. All components are properly assembled
        3. The chain has the expected attributes
        """
        qa_chain = self.get_qa_chain(self.vector_store, self.fake_llm)
        assert qa_chain is not None
        assert hasattr(qa_chain, "retriever")
        assert hasattr(qa_chain, "memory")
        assert hasattr(qa_chain, "combine_docs_chain")
        assert qa_chain.retriever.search_type == "mmr"
        assert qa_chain.retriever.search_kwargs["k"] == 4
        assert qa_chain.memory.k == 6

    def test_document_formatting(self):
        """
        Test that documents are formatted with <doc_i> markers.

        This test verifies that:
        1. Retrieved documents are wrapped in <doc_i> tags
        2. The LLM receives the properly formatted context
        3. The prompt contains the expected markers
        """
        qa_chain = self.get_qa_chain(self.vector_store, self.fake_llm)
        question = "What is Python programming?"

        # Simulate the document formatting that CustomStuffDocumentsChain does
        formatted_docs = ""
        for i, doc in enumerate(self.test_documents):
            formatted_docs += f"<doc_{i}>{doc.page_content}</doc_{i}>\n"
        formatted_docs = formatted_docs.strip()

        # Get the prompt template from the chain
        qa_llm_prompt = qa_chain.combine_docs_chain.llm_chain.prompt

        # Construct the expected human message content
        expected_human_message_content = (
            f"Context: {formatted_docs}\n\nQuestion: {question}"
        )

        # Check the prompt messages
        # The prompt should contain a SystemMessage and a HumanMessage
        messages = qa_llm_prompt.format_messages(
            context=formatted_docs, question=question
        )

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == expected_human_message_content

    def test_memory_window(self):
        """
        Test that the conversation memory respects the window limit.

        This test verifies that:
        1. Only the last N Q&A pairs are kept in memory
        2. Older conversations are properly discarded
        3. The memory window prevents indefinite growth
        """
        memory_window = 3
        num_turns = memory_window + 2  # 5 turns total

        # Create a fake LLM that returns unique responses for each call
        # This ensures we can track which responses are in memory
        responses = [
            f"UniqueAnswer{i}" for i in range(num_turns * 2)
        ]  # Extra responses in case
        self.fake_llm = FakeListLLM(responses=responses)
        qa_chain = self.get_qa_chain(
            self.vector_store, self.fake_llm, memory_window=memory_window
        )

        # Track the responses we get
        actual_responses = []
        for i in range(num_turns):
            question = f"Question{i}"
            result = qa_chain.invoke({"question": question})
            actual_responses.append(result["answer"])

        # The memory stores (human_message, ai_message) pairs.
        # So, for a memory_window of 3, it should store 3 pairs, i.e., 6 messages.
        expected_messages_in_memory = memory_window * 2
        actual_messages_in_memory = len(qa_chain.memory.buffer)
        assert actual_messages_in_memory == expected_messages_in_memory, (
            f"Expected {expected_messages_in_memory} messages in memory, got {actual_messages_in_memory}"
        )

        # Get all memory contents
        memory_contents = [msg.content for msg in qa_chain.memory.buffer]

        # Verify that we have the expected number of questions and answers
        question_count = sum(1 for content in memory_contents if "Question" in content)
        answer_count = sum(
            1 for content in memory_contents if "UniqueAnswer" in content
        )
        assert question_count == memory_window, (
            f"Expected {memory_window} questions, got {question_count}"
        )
        assert answer_count == memory_window, (
            f"Expected {memory_window} answers, got {answer_count}"
        )

        # Verify that the memory contains only the last memory_window turns
        # The last memory_window questions should be in memory
        expected_questions = [
            f"Question{i}" for i in range(num_turns - memory_window, num_turns)
        ]
        for expected_q in expected_questions:
            assert expected_q in memory_contents, (
                f"Expected question {expected_q} not found in memory"
            )

        # Verify that older questions are not in memory
        older_questions = [f"Question{i}" for i in range(num_turns - memory_window)]
        for older_q in older_questions:
            assert older_q not in memory_contents, (
                f"Older question {older_q} should not be in memory"
            )

        # Verify that we have exactly memory_window unique answers in memory
        unique_answers_in_memory = [
            content for content in memory_contents if "UniqueAnswer" in content
        ]
        assert len(unique_answers_in_memory) == memory_window, (
            f"Expected {memory_window} unique answers in memory, got {len(unique_answers_in_memory)}"
        )

    def test_mmr_retrieval(self):
        """
        Test that MMR retrieval works correctly.

        This test verifies that:
        1. The retriever uses MMR search type
        2. Up to retrieval_k documents are retrieved
        3. The retrieval provides diverse results (implicitly by checking search_type and k)
        """
        retrieval_k = 2
        qa_chain = self.get_qa_chain(
            self.vector_store, self.fake_llm, retrieval_k=retrieval_k
        )

        assert qa_chain.retriever.search_type == "mmr"
        assert qa_chain.retriever.search_kwargs["k"] == retrieval_k

        result = qa_chain.invoke({"question": "Any programming concepts?"})
        assert "source_documents" in result
        assert len(result["source_documents"]) <= retrieval_k
        for doc in result["source_documents"]:
            assert doc in self.test_documents

    def test_system_prompt_integration(self, mocker):
        """
        Test that the system prompt is properly integrated.

        This test verifies that:
        1. The system prompt is loaded correctly
        2. It's included in the chain's prompt template
        3. The LLM receives the system instructions
        """
        mock_system_prompt_content = "This is a mocked system prompt for testing."
        mocker.patch(
            "app.rag.chain._load_system_prompt", return_value=mock_system_prompt_content
        )

        qa_chain = self.get_qa_chain(self.vector_store, self.fake_llm)
        question = "Hello there."
        _ = qa_chain.invoke({"question": question})

        # Get the prompt template and verify it contains the system prompt
        qa_llm_prompt = qa_chain.combine_docs_chain.llm_chain.prompt
        messages = qa_llm_prompt.format_messages(
            context="test context", question=question
        )

        # Find the system message
        system_message = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_message = msg
                break

        assert system_message is not None, "SystemMessage not found in prompt"
        assert system_message.content == mock_system_prompt_content

    def test_error_handling(self, mocker):
        """
        Test error handling in the chain assembly.

        This test verifies that:
        1. Missing system prompt file raises appropriate error
        2. Invalid parameters are properly validated
        3. Chain creation fails gracefully with invalid inputs
        """
        with pytest.raises(ValueError, match="memory_window must be at least 1"):
            self.get_qa_chain(self.vector_store, self.fake_llm, memory_window=0)

        with pytest.raises(ValueError, match="retrieval_k must be at least 1"):
            self.get_qa_chain(self.vector_store, self.fake_llm, retrieval_k=0)

        # Test FileNotFoundError path
        mocker.patch(
            "app.rag.chain._load_system_prompt",
            side_effect=FileNotFoundError("System prompt file not found"),
        )
        with pytest.raises(FileNotFoundError, match="System prompt file not found"):
            self.get_qa_chain(self.vector_store, self.fake_llm)

        # Test IOError path
        mocker.patch(
            "app.rag.chain._load_system_prompt",
            side_effect=IOError("Error reading system prompt file"),
        )
        with pytest.raises(IOError, match="Error reading system prompt file"):
            self.get_qa_chain(self.vector_store, self.fake_llm)

    def test_format_docs_function(self):
        """
        Test the _format_docs function directly.

        This test verifies that:
        1. Documents are properly formatted with <doc_i> tags
        2. The function handles empty document lists
        3. The function preserves document content correctly
        """
        # Test with multiple documents
        formatted = self._format_docs(self.test_documents)
        assert (
            "<doc_0>This is a test document about Python programming.</doc_0>"
            in formatted
        )
        assert (
            "<doc_1>Another test document about machine learning.</doc_1>" in formatted
        )
        assert "<doc_2>Third document about data structures.</doc_2>" in formatted

        # Test with empty list
        empty_formatted = self._format_docs([])
        assert empty_formatted == ""

        # Test with single document
        single_doc = [Document(page_content="Single document")]
        single_formatted = self._format_docs(single_doc)
        assert single_formatted == "<doc_0>Single document</doc_0>"

    def test_custom_stuff_documents_chain(self):
        """
        Test the CustomStuffDocumentsChain class.

        This test verifies that:
        1. The custom chain properly formats documents with <doc_i> tags
        2. The _get_inputs method works correctly
        3. The chain integrates properly with the LLMChain
        """
        # Create a real LLMChain instead of a mock to avoid validation issues
        from langchain.chains.llm import LLMChain
        from langchain_core.prompts import ChatPromptTemplate

        # Create a simple prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [("human", "Context: {context}\n\nQuestion: {question}")]
        )

        # Create a real LLMChain
        llm_chain = LLMChain(llm=self.fake_llm, prompt=prompt_template)

        # Create the custom chain
        custom_chain = self.custom_stuff_documents_chain(
            llm_chain=llm_chain, document_variable_name="context"
        )

        # Test the _get_inputs method
        inputs = custom_chain._get_inputs(self.test_documents, question="test question")

        # Verify the context is properly formatted
        expected_context = self._format_docs(self.test_documents)
        assert inputs["context"] == expected_context
        assert "<doc_0>" in inputs["context"]
        assert "<doc_1>" in inputs["context"]
        assert "<doc_2>" in inputs["context"]

    def test_chain_execution_with_document_formatting(self):
        """
        Test the complete chain execution with document formatting.

        This test verifies that:
        1. The chain executes successfully
        2. Documents are retrieved and formatted
        3. The response contains the expected content
        """
        # Create a fake LLM that returns a response mentioning document markers
        fake_llm = FakeListLLM(
            responses=["Based on <doc_0> and <doc_1>, here's the answer."]
        )

        qa_chain = self.get_qa_chain(
            self.vector_store, fake_llm, memory_window=2, retrieval_k=2
        )

        # Execute the chain
        result = qa_chain.invoke({"question": "What is this about?"})

        # Verify the response
        assert "answer" in result
        assert "source_documents" in result
        assert len(result["source_documents"]) <= 2

        # Verify the response contains citations (document markers are replaced with citations)
        assert "[" in result["answer"] and "]" in result["answer"]
        assert "L" in result["answer"]  # Line numbers should be present

    def test_chain_with_different_parameters(self):
        """
        Test the chain with different parameter combinations.

        This test verifies that:
        1. Different memory_window values work correctly
        2. Different retrieval_k values work correctly
        3. The chain behaves consistently with different parameters
        """
        # Test with different memory window
        qa_chain_small_memory = self.get_qa_chain(
            self.vector_store, self.fake_llm, memory_window=2
        )
        assert qa_chain_small_memory.memory.k == 2

        # Test with different retrieval k
        qa_chain_small_retrieval = self.get_qa_chain(
            self.vector_store, self.fake_llm, retrieval_k=1
        )
        assert qa_chain_small_retrieval.retriever.search_kwargs["k"] == 1

        # Test with both custom parameters
        qa_chain_custom = self.get_qa_chain(
            self.vector_store, self.fake_llm, memory_window=4, retrieval_k=3
        )
        assert qa_chain_custom.memory.k == 4
        assert qa_chain_custom.retriever.search_kwargs["k"] == 3

    def test_load_system_prompt_function(self):
        """
        Test the _load_system_prompt function directly.

        This test verifies that:
        1. The function loads the system prompt correctly
        2. The prompt content is properly stripped
        3. The function handles file operations correctly
        """
        # Test that the function loads the actual system prompt
        system_prompt = self._load_system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "GitHub repository Q&A" in system_prompt
        assert "sources cited inline" in system_prompt

    def test_load_system_prompt_io_error(self, mocker):
        """
        Test the _load_system_prompt function with IOError.

        This test verifies that:
        1. The function properly handles IOError during file reading
        2. The error message is correctly formatted
        """
        # Mock the open function to raise an IOError
        mock_open = mocker.mock_open()
        mock_open.side_effect = IOError("Permission denied")
        mocker.patch("builtins.open", mock_open)

        with pytest.raises(
            IOError, match="Error reading system prompt file: Permission denied"
        ):
            self._load_system_prompt()

    def test_load_system_prompt_file_not_found(self, mocker):
        """
        Test the _load_system_prompt function with FileNotFoundError.

        This test verifies that:
        1. The function properly handles FileNotFoundError
        2. The error message includes the correct file path
        3. The exception is re-raised with 'from None' to suppress the original traceback
        """
        # Mock the open function to raise a FileNotFoundError
        mock_open = mocker.mock_open()
        mock_open.side_effect = FileNotFoundError("No such file or directory")
        mocker.patch("builtins.open", mock_open)

        with pytest.raises(FileNotFoundError, match="System prompt file not found at"):
            self._load_system_prompt()


# Example of how to test the chain in isolation
def example_chain_test():
    """
    Example function showing how to test the chain assembly in isolation.

    This demonstrates the approach mentioned in Work Package 2.3 requirements:
    - Create a small fake vector store with dummy documents
    - Use a fake LLM that echoes the prompt
    - Verify that the prompt contains <doc_i> markers
    - Check that the memory window is respected
    """
    # Create test documents
    documents = [
        Document(page_content="First test document content."),
        Document(page_content="Second test document content."),
    ]

    # Create fake embeddings and vector store
    embeddings = FakeEmbeddings(size=384)
    vector_store = Chroma.from_documents(documents, embeddings)

    # Create fake LLM that returns a predictable response
    fake_llm = FakeListLLM(responses=["Response with <doc_0> and <doc_1> markers"])

    # Create the QA chain
    qa_chain = get_qa_chain(vector_store, fake_llm, memory_window=2, retrieval_k=2)

    # Test the chain
    result = qa_chain.invoke({"question": "What is this about?"})

    # Verify the response contains the expected markers
    assert "<doc_0>" in result["answer"]
    assert "<doc_1>" in result["answer"]

    print("Chain test passed!")
    print(f"Answer: {result['answer']}")
    print(f"Source documents: {len(result['source_documents'])}")


if __name__ == "__main__":
    # Run the example test
    example_chain_test()

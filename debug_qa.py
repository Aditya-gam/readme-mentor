#!/usr/bin/env python3
"""
Debug script to test the QA system step by step.
"""

import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def test_qa_system():
    """Test the QA system step by step."""

    print("🔍 Debugging QA System")
    print("=" * 30)

    try:
        # Step 1: Test imports
        print("1. Testing imports...")
        from app.embeddings.ingest import get_embedding_model, get_vector_store
        from app.llm.provider import get_chat_model
        from app.rag.chain import HybridREADMERetriever, get_qa_chain

        print("✅ All imports successful")

        # Step 2: Test vector store loading
        print("\n2. Testing vector store loading...")
        repo_id = "google-gemini_gemini-cli"
        embedding_model = get_embedding_model()
        vector_store = get_vector_store(repo_id, embedding_model)
        print("✅ Vector store loaded successfully")

        # Step 3: Test LLM loading
        print("\n3. Testing LLM loading...")
        llm = get_chat_model()
        print("✅ LLM loaded successfully")

        # Step 4: Test retriever creation
        print("\n4. Testing retriever creation...")
        retriever = HybridREADMERetriever(vector_store, k=4)
        print("✅ Retriever created successfully")

        # Step 5: Test document retrieval
        print("\n5. Testing document retrieval...")
        test_query = "How do I run Gemini-CLI in my terminal?"
        docs = retriever.get_relevant_documents(test_query)
        print(f"✅ Retrieved {len(docs)} documents")

        # Step 6: Test QA chain creation
        print("\n6. Testing QA chain creation...")
        qa_chain = get_qa_chain(vector_store, llm)
        print("✅ QA chain created successfully")

        # Step 7: Test QA chain invocation
        print("\n7. Testing QA chain invocation...")
        result = qa_chain.invoke({"question": test_query, "chat_history": []})
        print("✅ QA chain invoked successfully")
        print(f"Answer: {result.get('answer', 'No answer')[:100]}...")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("OPENAI_API_KEY", "test")

    success = test_qa_system()
    sys.exit(0 if success else 1)

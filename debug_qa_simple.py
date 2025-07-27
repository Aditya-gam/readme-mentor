#!/usr/bin/env python3
"""
Simple debug script to test the QA system without circular imports.
"""

import sys
from pathlib import Path


def test_qa_system():
    """Test the QA system step by step."""

    print("🔍 Debugging QA System (Simple)")
    print("=" * 35)

    try:
        # Step 1: Test basic imports
        print("1. Testing basic imports...")
        print("✅ Basic imports successful")

        # Step 2: Test our custom retriever
        print("\n2. Testing custom retriever...")
        sys.path.insert(0, str(Path(__file__).parent / "app"))

        # Import without triggering the circular import
        import app.rag.chain

        retriever_class = app.rag.chain.READMEPrioritizingRetriever
        print("✅ Custom retriever imported successfully")

        # Step 3: Test retriever instantiation (mock)
        print("\n3. Testing retriever instantiation...")
        # Create a mock vector store for testing

        class MockVectorStore:
            def as_retriever(self, **kwargs):
                class MockRetriever:
                    def get_relevant_documents(self, query):
                        return []

                return MockRetriever()

        mock_vector_store = MockVectorStore()
        retriever = retriever_class(mock_vector_store, k=4)
        print("✅ Retriever instantiated successfully")

        # Step 4: Test document retrieval
        print("\n4. Testing document retrieval...")
        docs = retriever.get_relevant_documents("test query")
        print(f"✅ Retrieved {len(docs)} documents")

        print("\n✅ All basic tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qa_system()
    sys.exit(0 if success else 1)

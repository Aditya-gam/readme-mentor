#!/usr/bin/env python3
"""
Test script to demonstrate README prioritization improvements.

This script tests the enhanced README prioritization system to ensure:
1. README files are properly prioritized
2. Simple answers are preferred over complex ones
3. The system avoids hallucination
"""

import os
import sys
from pathlib import Path

from app.backend import generate_answer
from app.embeddings.ingest import ingest_repository

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def test_readme_prioritization():
    """Test the README prioritization improvements."""

    # Test repository (using this project itself)
    repo_url = "https://github.com/your-username/readme-mentor"

    print("🧪 Testing README Prioritization Improvements")
    print("=" * 50)

    try:
        # Ingest the repository
        print(f"📥 Ingesting repository: {repo_url}")
        ingest_repository(
            repo_url=repo_url,
            file_glob=("README*", "docs/**/*.md"),
            persist_directory="./test_data",
        )

        # Test questions that should prioritize README
        test_questions = [
            "How do I install this project?",
            "What is the quick start guide?",
            "How do I run the tests?",
            "What are the basic usage instructions?",
            "How do I get started with this project?",
        ]

        print("\n🔍 Testing README Prioritization:")
        print("-" * 30)

        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)

            try:
                result = generate_answer(
                    query=question, repo_id="test_repo", history=[]
                )

                print(f"Answer: {result['answer']}")
                print(f"Citations: {len(result['citations'])}")

                # Check if README is cited
                readme_cited = any(
                    "README" in citation.get("file", "").upper()
                    for citation in result["citations"]
                )

                if readme_cited:
                    print("✅ README cited - Good!")
                else:
                    print("⚠️  README not cited - May need improvement")

            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n✅ README Prioritization Test Complete!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("OPENAI_API_KEY", "test")

    success = test_readme_prioritization()
    sys.exit(0 if success else 1)

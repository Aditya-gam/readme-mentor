#!/usr/bin/env python3
"""
README-Mentor CLI Demo

This script runs an actual CLI session with a real repository to demonstrate
the enhanced CLI functionality with real ingestion and Q&A.

Usage:
    python demo.py
"""

import subprocess
import sys
import time


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)


def run_actual_demo():
    """Run an actual CLI session with a real repository."""
    print_header("README-Mentor CLI Demo")

    print("This demo will run an actual Q&A session with a real repository.")
    print("Repository: https://github.com/google-gemini/gemini-cli.git")
    print()
    print("You will see:")
    print("• Actual repository ingestion process")
    print("• Real Q&A responses based on repository content")
    print("• Source citations and response times")
    print("• Interactive session features")
    print()

    # Questions to ask about Gemini CLI
    questions = [
        "What is Gemini CLI and what are its main features?",
        "How do I install and get started with Gemini CLI?",
        "What are the authentication options for Gemini CLI?",
        "What are some popular tasks you can do with Gemini CLI?",
    ]

    try:
        print("🚀 Starting actual CLI session...")
        print(
            "Command: ./readme-mentor qa https://github.com/google-gemini/gemini-cli.git --fast"
        )
        print("\n" + "=" * 60)

        # Start the CLI process
        process = subprocess.Popen(
            [
                "./readme-mentor",
                "qa",
                "https://github.com/google-gemini/gemini-cli.git",
                "--fast",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Wait for the session to start and ingestion to complete
        print("⏳ Waiting for repository ingestion and session start...")

        # Send questions one by one
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")
            print("🤔 Thinking...")

            # Send the question
            process.stdin.write(question + "\n")
            process.stdin.flush()

            # Wait a bit for processing
            time.sleep(2)

            # Read response
            response = process.stdout.readline()
            while response and not response.strip().startswith("❓ Question:"):
                if response.strip():
                    print(response.strip())
                response = process.stdout.readline()

            print("-" * 60)
            time.sleep(1)

        # End the session
        print("\n❓ Question: quit")
        process.stdin.write("quit\n")
        process.stdin.flush()

        # Wait for session to end
        process.wait()
        print("👋 Session ended")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

    return True


def main():
    """Run the demo."""
    print("🚀 README-Mentor CLI Demo")
    print(
        "This demo will run an actual Q&A session with the Google Gemini CLI repository"
    )

    try:
        print_header("Demo Information")
        print("This demo will:")
        print("1. Start an actual CLI session")
        print("2. Ingest the Google Gemini CLI repository")
        print("3. Ask 4 real questions about Gemini CLI")
        print("4. Show actual responses with citations")
        print("5. End the session")
        print()
        print("⚠️  Note: Repository ingestion may take several minutes.")
        print()

        success = run_actual_demo()
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n⚠️  Demo encountered issues.")

        print_header("Demo Complete")
        print("🎉 You've seen the enhanced CLI in action!")
        print("\n💡 To use README-Mentor with any repository:")
        print("   ./readme-mentor qa <your-repo-url>")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

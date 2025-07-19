#!/usr/bin/env python3
"""Script to investigate the structure of test repositories."""

import logging
import sys
from pathlib import Path

from app.github.loader import fetch_repository_files
from app.utils.validators import validate_repo_url

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Test repositories
TEST_REPOSITORIES = {
    "flask": "https://github.com/pallets/flask",
    "black": "https://github.com/psf/black",
    "hello_world": "https://github.com/octocat/Hello-World",
    "test_tlb": "https://github.com/torvalds/test-tlb",
    "kkomiyama": "https://github.com/kkomiyama/kkomiyama.github.io",
    "codedocs": "https://github.com/CodeDocs/CodeDocs",
    "no_readme": "https://github.com/gittertestbot/no-readme-markdown-file-2",
}


def investigate_repository_structure():
    """Investigate the actual structure of each repository."""
    print("Repository Structure Investigation")
    print("=" * 50)

    results = {}

    for repo_name, url in TEST_REPOSITORIES.items():
        print(f"\n{'='*60}")
        print(f"Investigating: {repo_name}")
        print(f"URL: {url}")
        print(f"{'='*60}")

        try:
            # Validate URL first
            validated_url = validate_repo_url(url)
            print(f"✅ URL validation passed: {validated_url}")

            # Fetch files
            saved_files = fetch_repository_files(validated_url)

            # Analyze structure
            readme_files = [f for f in saved_files if "README" in f]
            docs_files = [f for f in saved_files if "docs/" in f]
            all_md_files = [f for f in saved_files if f.endswith(
                ('.md', '.markdown', '.mdx'))]

            result = {
                "total_files": len(saved_files),
                "readme_files": readme_files,
                "docs_files": docs_files,
                "all_md_files": all_md_files,
                "has_readme": len(readme_files) > 0,
                "has_docs": len(docs_files) > 0,
                "file_paths": saved_files,
            }

            results[repo_name] = result

            # Print results
            print(f"📊 Results for {repo_name}:")
            print(f"   - Total files saved: {len(saved_files)}")
            print(f"   - README files: {len(readme_files)}")
            print(f"   - Docs files: {len(docs_files)}")
            print(f"   - All markdown files: {len(all_md_files)}")

            if saved_files:
                print("   - Files saved:")
                for file_path in saved_files:
                    file_size = Path(file_path).stat().st_size if Path(
                        file_path).exists() else 0
                    print(f"     * {file_path} ({file_size} bytes)")

            # Check if files exist on disk
            print("   - Files on disk:")
            for file_path in saved_files:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"     ✅ {file_path} ({file_size} bytes)")
                else:
                    print(f"     ❌ {file_path} (not found)")

        except Exception as e:
            print(f"❌ Error investigating {repo_name}: {e}")
            results[repo_name] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for repo_name, result in results.items():
        if "error" in result:
            print(f"❌ {repo_name}: {result['error']}")
        else:
            print(f"✅ {repo_name}: {result['total_files']} files")
            if result['has_readme']:
                print(f"   - Has README: {result['readme_files']}")
            if result['has_docs']:
                print(f"   - Has docs: {result['docs_files']}")

    return results


def suggest_updated_expectations(results):
    """Suggest updated expectations based on actual results."""
    print(f"\n{'='*60}")
    print("SUGGESTED UPDATED EXPECTATIONS")
    print(f"{'='*60}")

    expectations = {}

    for repo_name, result in results.items():
        if "error" in result:
            continue

        expectation = {
            "has_readme": result["has_readme"],
            "has_docs": result["has_docs"],
            "min_files": result["total_files"],
            "actual_files": result["total_files"],
        }

        expectations[repo_name] = expectation

        print(f"{repo_name}:")
        print(f"  has_readme: {result['has_readme']}")
        print(f"  has_docs: {result['has_docs']}")
        print(f"  min_files: {result['total_files']}")
        print(f"  actual_files: {result['total_files']}")
        print()

    return expectations


if __name__ == "__main__":
    results = investigate_repository_structure()
    expectations = suggest_updated_expectations(results)

    print("Investigation completed!")

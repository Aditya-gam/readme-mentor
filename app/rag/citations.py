import re
from typing import Dict, List

from langchain.schema import Document


def render_citations(answer_text: str, source_docs: List[Document]) -> str:
    """
    Post-processes the LLM's answer to replace placeholder tokens with actual citations.
    Limits the number of unique citations to at most three and truncates the answer.

    Args:
        answer_text: The LLM's answer text containing placeholder tokens like <doc_0>.
        source_docs: A list of LangChain Document objects, corresponding to the source documents.

    Returns:
        The processed answer text with inline markdown citations.

    Raises:
        ValueError: If answer_text is None or empty.
        TypeError: If source_docs is not a list.
    """
    # Input validation
    if not answer_text:
        return ""

    if not isinstance(source_docs, list):
        raise TypeError("source_docs must be a list of Document objects")

    # Find all opening and closing doc placeholders
    opening_pattern = r"<doc_(\d+)>"
    closing_pattern = r"</doc_(\d+)>"

    opening_matches = re.findall(opening_pattern, answer_text)
    _ = re.findall(closing_pattern, answer_text)

    # Get unique document indices from opening tags
    doc_indices = set(int(idx) for idx in opening_matches)

    # Build citations dictionary
    citations: Dict[int, str] = {}
    for doc_idx in doc_indices:
        if doc_idx < len(source_docs):
            doc = source_docs[doc_idx]
            metadata = doc.metadata

            file_name = metadata.get("file")
            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")

            # Validate metadata fields
            if file_name and start_line is not None and end_line is not None:
                try:
                    start_line = int(start_line)
                    end_line = int(end_line)
                    citation_string = f"[{file_name} L{start_line}–{end_line}]"
                    citations[doc_idx] = citation_string
                except (ValueError, TypeError):
                    # Skip invalid line numbers
                    continue

    # Limit to at most three unique citations
    # Assuming source_docs are ordered by relevance (doc_0 most relevant)
    sorted_doc_indices = sorted(citations.keys())
    if len(sorted_doc_indices) > 3:
        # Remove citations for documents beyond the first 3
        for doc_idx in sorted_doc_indices[3:]:
            # Remove both opening and closing tags for this document
            answer_text = re.sub(rf"<doc_{doc_idx}>", "", answer_text)
            answer_text = re.sub(rf"</doc_{doc_idx}>", "", answer_text)

        # Keep only the first three citations
        citations = {k: citations[k] for k in sorted_doc_indices[:3]}

    # Replace opening placeholders with actual citations
    for doc_idx, citation_string in citations.items():
        answer_text = re.sub(rf"<doc_{doc_idx}>", citation_string, answer_text)

    # Remove all remaining closing tags (for both cited and non-cited documents)
    answer_text = re.sub(r"</doc_\d+>", "", answer_text)

    # Clean up any remaining malformed placeholders
    answer_text = re.sub(r"<doc_[^>]*>", "", answer_text)

    # Trim trailing whitespace and newline characters
    answer_text = answer_text.strip()

    # Truncate to ~120 words
    words = answer_text.split()
    if len(words) > 120:
        answer_text = " ".join(words[:120]) + "..."

    return answer_text


def _validate_metadata(metadata: dict) -> bool:
    """
    Validate that metadata contains required fields for citation generation.

    Args:
        metadata: Document metadata dictionary

    Returns:
        True if metadata is valid, False otherwise
    """
    required_fields = ["file", "start_line", "end_line"]
    return all(
        field in metadata and metadata[field] is not None for field in required_fields
    )


def _count_words(text: str) -> int:
    """
    Count the number of words in a text string.

    Args:
        text: Input text string

    Returns:
        Number of words in the text
    """
    return len(text.split())

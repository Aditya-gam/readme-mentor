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

    # Extract document indices and build citations
    doc_indices = _extract_document_indices(answer_text)
    citations = _build_citations(doc_indices, source_docs)

    # Limit citations and process answer
    citations = _limit_citations(citations, answer_text)
    answer_text = _replace_placeholders(answer_text, citations)
    answer_text = _cleanup_placeholders(answer_text)

    # Final processing
    answer_text = answer_text.strip()
    return _truncate_answer(answer_text)


def _extract_document_indices(answer_text: str) -> set[int]:
    """Extract unique document indices from opening tags."""
    opening_pattern = r"<doc_(\d+)>"
    opening_matches = re.findall(opening_pattern, answer_text)
    return {int(idx) for idx in opening_matches}


def _build_citations(
    doc_indices: set[int], source_docs: List[Document]
) -> Dict[int, str]:
    """Build citations dictionary from document indices."""
    citations: Dict[int, str] = {}
    for doc_idx in doc_indices:
        if doc_idx < len(source_docs):
            doc = source_docs[doc_idx]
            metadata = doc.metadata

            # Try both field naming conventions
            file_name = metadata.get("file") or metadata.get("source")
            start_line = metadata.get("start_line") or metadata.get("line_start")
            end_line = metadata.get("end_line") or metadata.get("line_end")

            if _validate_metadata_fields(file_name, start_line, end_line):
                try:
                    start_line = int(start_line)
                    end_line = int(end_line)
                    citation_string = f"[{file_name} L{start_line}–{end_line}]"
                    citations[doc_idx] = citation_string
                except (ValueError, TypeError):
                    continue
    return citations


def _validate_metadata_fields(file_name, start_line, end_line) -> bool:
    """Validate that metadata fields are present and not None."""
    return file_name and start_line is not None and end_line is not None


def _limit_citations(citations: Dict[int, str], answer_text: str) -> Dict[int, str]:
    """Limit citations to at most three and remove excess placeholders."""
    sorted_doc_indices = sorted(citations.keys())
    if len(sorted_doc_indices) > 3:
        # Remove citations for documents beyond the first 3
        for doc_idx in sorted_doc_indices[3:]:
            answer_text = re.sub(rf"<doc_{doc_idx}>", "", answer_text)
            answer_text = re.sub(rf"</doc_{doc_idx}>", "", answer_text)
        # Keep only the first three citations
        citations = {k: citations[k] for k in sorted_doc_indices[:3]}
    return citations


def _replace_placeholders(answer_text: str, citations: Dict[int, str]) -> str:
    """Replace opening placeholders with actual citations."""
    for doc_idx, citation_string in citations.items():
        answer_text = re.sub(rf"<doc_{doc_idx}>", citation_string, answer_text)
    return answer_text


def _cleanup_placeholders(answer_text: str) -> str:
    """Clean up remaining closing tags and malformed placeholders."""
    # Remove all remaining closing tags
    answer_text = re.sub(r"</doc_\d+>", "", answer_text)
    # Clean up any remaining malformed placeholders
    answer_text = re.sub(r"<doc_[^>]*>", "", answer_text)
    return answer_text


def _truncate_answer(answer_text: str) -> str:
    """Truncate answer to ~200 words to allow for more complete answers."""
    words = answer_text.split()
    if len(words) > 200:
        answer_text = " ".join(words[:200]) + "..."
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

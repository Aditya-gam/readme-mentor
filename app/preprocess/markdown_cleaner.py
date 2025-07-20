"""Markdown content cleaning and normalization utilities.

This module provides functionality to clean raw Markdown text by removing
unwanted sections while preserving line numbers for citation purposes.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


@dataclass
class CleanedDocument:
    """Represents a cleaned Markdown document with line offset mapping.

    This class holds the cleaned text content along with metadata and
    line offset information for citation purposes.

    Attributes:
        text: The cleaned text content
        path: Optional path to the original file
        line_offsets: List of character offsets for each line start
        metadata: Additional processing metadata
    """

    text: str
    path: Optional[Path]
    line_offsets: List[int]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate and compute line offsets if not provided."""
        if not self.line_offsets:
            self.line_offsets = self._compute_line_offsets()

    def _compute_line_offsets(self) -> List[int]:
        """Compute character offsets for the start of each line.

        Returns:
            List where each element is the character offset of the start
            of the corresponding line (0-indexed).
        """
        offsets = [0]  # First line always starts at offset 0
        current_offset = 0

        for char in self.text:
            current_offset += 1
            if char == "\n":
                offsets.append(current_offset)

        return offsets

    def get_line_range(self, start_char: int, end_char: int) -> Tuple[int, int]:
        """Get the line range corresponding to a character range.

        Args:
            start_char: Starting character offset (inclusive)
            end_char: Ending character offset (exclusive)

        Returns:
            Tuple of (start_line, end_line) where both are 0-indexed

        Raises:
            ValueError: If start_char or end_char are invalid
        """
        if start_char < 0 or end_char < 0:
            raise ValueError("Character offsets must be non-negative")
        if start_char >= end_char:
            raise ValueError("start_char must be less than end_char")
        if end_char > len(self.text):
            raise ValueError("end_char exceeds text length")

        # Find the line containing start_char
        start_line = bisect_right(self.line_offsets, start_char) - 1
        if start_line < 0:
            start_line = 0

        # Find the line containing end_char
        end_line = bisect_right(self.line_offsets, end_char) - 1
        if end_line < 0:
            end_line = 0

        return start_line, end_line

    def get_line_content(self, line_number: int) -> str:
        """Get the content of a specific line.

        Args:
            line_number: 0-indexed line number

        Returns:
            Content of the specified line

        Raises:
            IndexError: If line_number is out of range
        """
        if line_number < 0 or line_number >= len(self.line_offsets):
            raise IndexError(f"Line number {line_number} out of range")

        start_offset = self.line_offsets[line_number]
        if line_number + 1 < len(self.line_offsets):
            end_offset = self.line_offsets[line_number + 1]
        else:
            end_offset = len(self.text)

        return self.text[start_offset:end_offset].rstrip("\n")


class LineOffsetMapper:
    """Maps character offsets to line numbers in text content.

    This class provides efficient mapping between character positions
    and line numbers using binary search for optimal performance.
    """

    def __init__(self, text: str):
        """Initialize the mapper with text content.

        Args:
            text: The text content to map
        """
        self.text = text
        self.line_offsets = self._compute_line_offsets()

    def _compute_line_offsets(self) -> List[int]:
        """Compute character offsets for the start of each line.

        Returns:
            List where each element is the character offset of the start
            of the corresponding line (0-indexed).
        """
        offsets = [0]  # First line always starts at offset 0
        current_offset = 0

        for char in self.text:
            current_offset += 1
            if char == "\n":
                offsets.append(current_offset)

        return offsets

    def get_line_range(self, start_char: int, end_char: int) -> Tuple[int, int]:
        """Get the line range corresponding to a character range.

        Args:
            start_char: Starting character offset (inclusive)
            end_char: Ending character offset (exclusive)

        Returns:
            Tuple of (start_line, end_line) where both are 0-indexed

        Raises:
            ValueError: If start_char or end_char are invalid
        """
        if start_char < 0 or end_char < 0:
            raise ValueError("Character offsets must be non-negative")
        if start_char >= end_char:
            raise ValueError("start_char must be less than end_char")
        if end_char > len(self.text):
            raise ValueError("end_char exceeds text length")

        # Find the line containing start_char
        start_line = bisect_right(self.line_offsets, start_char) - 1
        if start_line < 0:
            start_line = 0

        # Find the line containing end_char
        end_line = bisect_right(self.line_offsets, end_char) - 1
        if end_line < 0:
            end_line = 0

        return start_line, end_line

    def get_line_number(self, char_offset: int) -> int:
        """Get the line number containing a character offset.

        Args:
            char_offset: Character offset to find line for

        Returns:
            0-indexed line number

        Raises:
            ValueError: If char_offset is invalid
        """
        if char_offset < 0:
            raise ValueError("Character offset must be non-negative")
        if char_offset > len(self.text):
            raise ValueError("Character offset exceeds text length")

        line_number = bisect_right(self.line_offsets, char_offset) - 1
        return max(0, line_number)


def detect_encoding(file_bytes: bytes) -> str:
    """Detect the encoding of file bytes.

    Args:
        file_bytes: Raw file bytes to analyze

    Returns:
        Detected encoding string (e.g., 'utf-8', 'latin-1')

    Raises:
        UnicodeDecodeError: If no encoding can be detected
    """
    # Try UTF-8 first as it's most common
    try:
        file_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # Use chardet to detect encoding
    result = chardet.detect(file_bytes)
    if result["confidence"] > 0.7 and result["encoding"]:
        try:
            file_bytes.decode(result["encoding"])
            return result["encoding"]
        except (UnicodeDecodeError, LookupError):
            pass

    # Fallback to latin-1 (never fails)
    return "latin-1"


def remove_html_comments(content: str) -> str:
    """Remove HTML comments from content while preserving line structure.

    Args:
        content: Input string that may contain HTML comments

    Returns:
        String with HTML comments replaced by empty lines
    """
    # Pattern to match HTML comments: <!-- ... -->
    comment_pattern = r"<!--.*?-->"

    def replace_comment(match: re.Match) -> str:
        """Replace comment with empty lines to preserve line count."""
        comment_text = match.group(0)
        # Count newlines in the comment and replace with same number of empty lines
        newline_count = comment_text.count("\n")
        return "\n" * newline_count if newline_count > 0 else ""

    return re.sub(comment_pattern, replace_comment, content, flags=re.DOTALL)


def remove_yaml_front_matter(content: str) -> str:
    """Remove YAML front matter from the beginning of content.

    Args:
        content: Input string that may contain YAML front matter

    Returns:
        String with YAML front matter removed, preserving line structure
    """
    lines = content.split("\n")

    # Check if content starts with YAML front matter
    if not lines or not lines[0].strip().startswith("---"):
        return content

    # Find the end of YAML front matter
    end_index = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_index = i
            break

    if end_index == -1:
        # No closing --- found, return original content
        return content

    # Remove YAML front matter and preserve line structure
    remaining_lines = lines[end_index + 1 :]
    return "\n" * (end_index + 1) + "\n".join(remaining_lines)


def remove_fenced_code_blocks(content: str, include_code: bool = False) -> str:
    """Remove fenced code blocks while preserving line structure.

    Args:
        content: Input string that may contain fenced code blocks
        include_code: If True, preserve code blocks; if False, remove them

    Returns:
        String with code blocks removed (if include_code=False) or preserved
    """
    if include_code:
        return content

    lines = content.split("\n")
    result_lines = []
    in_code_block = False
    code_fence_pattern = r"^```\w*$"

    for line in lines:
        if re.match(code_fence_pattern, line.strip()):
            # Toggle code block state
            in_code_block = not in_code_block
            # Replace fence line with empty line to preserve structure
            result_lines.append("")
        elif in_code_block:
            # Inside code block, replace with empty line
            result_lines.append("")
        else:
            # Outside code block, keep original line
            result_lines.append(line)

    return "\n".join(result_lines)


def _is_indented_code_line(line: str) -> bool:
    """Check if a line is indented code (4+ spaces or tab).

    Args:
        line: Line to check

    Returns:
        True if line is indented code, False otherwise
    """
    stripped = line.strip()
    return (
        (line.startswith("    ") or line.startswith("\t"))
        and stripped != ""
        # Don't treat indented comments as code
        and not line.startswith("    #")
    )


def _process_line_in_code_block(
    line: str, stripped: str, is_indented_code: bool
) -> tuple[str, bool]:
    """Process a line that's inside a code block.

    Args:
        line: Current line
        stripped: Stripped version of the line
        is_indented_code: Whether the line is indented code

    Returns:
        Tuple of (processed_line, new_in_code_block_state)
    """
    if stripped == "" or not is_indented_code:
        # End of code block
        return line, False
    # Inside code block, replace with empty line
    return "", True


def remove_indented_code_blocks(content: str, include_code: bool = False) -> str:
    """Remove indented code blocks while preserving line structure.

    Args:
        content: Input string that may contain indented code blocks
        include_code: If True, preserve code blocks; if False, remove them

    Returns:
        String with indented code blocks removed (if include_code=False) or preserved
    """
    if include_code:
        return content

    lines = content.split("\n")
    result_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        is_indented_code = _is_indented_code_line(line)

        if is_indented_code and not in_code_block:
            in_code_block = True

        if in_code_block:
            processed_line, in_code_block = _process_line_in_code_block(
                line, stripped, is_indented_code
            )
            result_lines.append(processed_line)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def markdown_to_plain_text(content: str) -> str:
    """Convert Markdown content to plain text while preserving line structure.

    Args:
        content: Markdown content to convert

    Returns:
        Plain text with preserved line structure
    """
    # Initialize markdown parser
    md = MarkdownIt()

    # Convert markdown to HTML
    html_content = md.render(content)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Get text content
    text_content = soup.get_text()

    # Preserve line structure by handling line breaks
    # Replace <br> tags with newlines and normalize
    text_content = text_content.replace("\r\n", "\n").replace("\r", "\n")

    return text_content


def clean_markdown_content(
    file_path: Optional[Path] = None,
    content: Optional[str] = None,
    include_code: bool = False,
) -> CleanedDocument:
    """Clean and normalize Markdown content.

    This function performs comprehensive cleaning of Markdown content:
    - Detects and handles file encoding
    - Removes HTML comments
    - Removes YAML front matter
    - Removes code blocks (unless include_code=True)
    - Converts to plain text while preserving line structure

    Args:
        file_path: Path to the Markdown file to process
        content: Raw Markdown content string (alternative to file_path)
        include_code: Whether to preserve code blocks in output

    Returns:
        CleanedDocument object containing:
        - text: The cleaned text content
        - path: Path to the original file (if provided)
        - line_offsets: Character offsets for line starts
        - metadata: Processing metadata including:
          - original_line_count: Number of lines in original content
          - cleaned_line_count: Number of lines in cleaned content
          - encoding: Detected encoding
          - processing_info: Additional processing details

    Raises:
        ValueError: If neither file_path nor content is provided
        FileNotFoundError: If file_path is provided but file doesn't exist
        UnicodeDecodeError: If file cannot be decoded with any encoding
    """
    if file_path is None and content is None:
        raise ValueError("Either file_path or content must be provided")

    metadata = {
        "original_line_count": 0,
        "cleaned_line_count": 0,
        "encoding": "utf-8",
        "processing_info": {},
    }

    # Read content from file if file_path provided
    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_bytes = file_path.read_bytes()
        encoding = detect_encoding(file_bytes)
        content = file_bytes.decode(encoding)
        metadata["encoding"] = encoding

    if content is None:
        raise ValueError("Content could not be read")

    # Store original line count
    original_lines = content.split("\n")
    metadata["original_line_count"] = len(original_lines)

    # Step 1: Remove HTML comments
    content = remove_html_comments(content)

    # Step 2: Remove YAML front matter
    content = remove_yaml_front_matter(content)

    # Step 3: Remove fenced code blocks
    content = remove_fenced_code_blocks(content, include_code)

    # Step 4: Remove indented code blocks
    content = remove_indented_code_blocks(content, include_code)

    # Step 5: Convert to plain text
    cleaned_text = markdown_to_plain_text(content)

    # Store cleaned line count
    cleaned_lines = cleaned_text.split("\n")
    metadata["cleaned_line_count"] = len(cleaned_lines)

    # Add processing info
    metadata["processing_info"] = {
        "html_comments_removed": True,
        "yaml_front_matter_removed": True,
        "code_blocks_removed": not include_code,
        "converted_to_plain_text": True,
    }

    # Create and return CleanedDocument
    return CleanedDocument(
        text=cleaned_text,
        path=file_path,
        line_offsets=[],  # Will be computed in __post_init__
        metadata=metadata,
    )


def clean_markdown_file(file_path: Path, include_code: bool = False) -> CleanedDocument:
    """Convenience function to clean a Markdown file.

    Args:
        file_path: Path to the Markdown file to process
        include_code: Whether to preserve code blocks in output

    Returns:
        CleanedDocument object containing cleaned text and metadata
    """
    return clean_markdown_content(file_path=file_path, include_code=include_code)


def clean_markdown_string(content: str, include_code: bool = False) -> CleanedDocument:
    """Convenience function to clean a Markdown string.

    Args:
        content: Raw Markdown content string
        include_code: Whether to preserve code blocks in output

    Returns:
        CleanedDocument object containing cleaned text and metadata
    """
    return clean_markdown_content(content=content, include_code=include_code)

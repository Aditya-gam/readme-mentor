"""Edge case tests for markdown cleaning functionality."""

import pytest

from app.preprocess.markdown_cleaner import (
    CleanedDocument,
    LineOffsetMapper,
    clean_markdown_content,
    clean_markdown_file,
    clean_markdown_string,
    markdown_to_plain_text,
    remove_fenced_code_blocks,
    remove_html_comments,
    remove_indented_code_blocks,
    remove_yaml_front_matter,
)


class TestLineOffsetMapperEdgeCases:
    """Test edge cases for LineOffsetMapper."""

    def test_line_offset_mapper_empty_text(self):
        """Test LineOffsetMapper with empty text."""
        mapper = LineOffsetMapper("")
        assert mapper.get_line_number(0) == 0
        # Should raise ValueError for out of bounds
        with pytest.raises(ValueError, match="Character offset exceeds text length"):
            mapper.get_line_number(10)

    def test_line_offset_mapper_single_character(self):
        """Test LineOffsetMapper with single character."""
        mapper = LineOffsetMapper("a")
        assert mapper.get_line_number(0) == 0
        assert mapper.get_line_number(1) == 0

    def test_line_offset_mapper_invalid_offsets(self):
        """Test LineOffsetMapper with invalid offsets."""
        mapper = LineOffsetMapper("Hello\nWorld")

        # Test negative offset
        with pytest.raises(ValueError, match="Character offset must be non-negative"):
            mapper.get_line_number(-1)

        # Test offset beyond text length
        with pytest.raises(ValueError, match="Character offset exceeds text length"):
            mapper.get_line_number(20)

    def test_line_offset_mapper_edge_line_numbers(self):
        """Test LineOffsetMapper with edge line numbers."""
        text = "Line 1\nLine 2\nLine 3"
        mapper = LineOffsetMapper(text)

        # Test start of each line
        assert mapper.get_line_number(0) == 0  # Start of first line
        assert mapper.get_line_number(7) == 1  # Start of second line
        assert mapper.get_line_number(14) == 2  # Start of third line

    def test_line_offset_mapper_text_without_newlines(self):
        """Test LineOffsetMapper with text without newlines."""
        text = "This is a single line of text without any newlines"
        mapper = LineOffsetMapper(text)

        # All offsets should map to line 0
        for i in range(len(text)):
            assert mapper.get_line_number(i) == 0


class TestCleanedDocumentEdgeCases:
    """Test edge cases for CleanedDocument."""

    def test_cleaned_document_empty_text(self):
        """Test CleanedDocument with empty text."""
        doc = CleanedDocument("", None, [], {})

        # Test line range with empty text
        with pytest.raises(ValueError, match="start_char must be less than end_char"):
            doc.get_line_range(0, 0)

        # Test line content
        assert doc.get_line_content(0) == ""

    def test_cleaned_document_single_line(self):
        """Test CleanedDocument with single line."""
        doc = CleanedDocument("Hello World", None, [], {})

        # Test line range
        start_line, end_line = doc.get_line_range(0, 11)
        assert start_line == 0
        assert end_line == 0

        # Test line content
        assert doc.get_line_content(0) == "Hello World"

    def test_cleaned_document_invalid_line_number(self):
        """Test CleanedDocument with invalid line number."""
        doc = CleanedDocument("Line 1\nLine 2", None, [], {})

        # Test negative line number
        with pytest.raises(IndexError, match="Line number -1 out of range"):
            doc.get_line_content(-1)

        # Test line number beyond text
        with pytest.raises(IndexError, match="Line number 10 out of range"):
            doc.get_line_content(10)

    def test_cleaned_document_last_line(self):
        """Test CleanedDocument with last line."""
        doc = CleanedDocument("Line 1\nLine 2", None, [], {})

        # Test last line
        assert doc.get_line_content(1) == "Line 2"

    def test_cleaned_document_line_with_newline(self):
        """Test CleanedDocument with line containing newline."""
        doc = CleanedDocument("Line 1\nLine 2\n", None, [], {})

        # Test line content
        assert doc.get_line_content(0) == "Line 1"
        assert doc.get_line_content(1) == "Line 2"


class TestCleanMarkdownContentEdgeCases:
    """Test edge cases for clean_markdown_content."""

    def test_clean_markdown_content_none_input(self):
        """Test clean_markdown_content with None input."""
        with pytest.raises(
            ValueError, match="Either file_path or content must be provided"
        ):
            clean_markdown_content()

    def test_clean_markdown_content_both_inputs(self):
        """Test clean_markdown_content with both inputs."""
        with pytest.raises(FileNotFoundError, match="File not found: test.md"):
            clean_markdown_content(content="test", file_path="test.md")

    def test_clean_markdown_content_empty_string(self):
        """Test clean_markdown_content with empty string."""
        result = clean_markdown_content(content="")
        assert result.text == ""

    def test_clean_markdown_content_whitespace_only(self):
        """Test clean_markdown_content with whitespace only."""
        result = clean_markdown_content(content="   \n\t\n  ")
        assert result.text.strip() == ""

    def test_clean_markdown_content_none_after_cleaning(self):
        """Test clean_markdown_content when cleaning results in None."""
        result = clean_markdown_content(content="<!-- Comment -->")
        assert result.text == ""

    def test_clean_markdown_content_with_code_included(self):
        """Test clean_markdown_content with code included."""
        content = """# Header
```python
def hello():
    print("Hello")
```
More content."""

        result = clean_markdown_content(content=content, include_code=True)
        assert "def hello():" in result.text
        assert "More content." in result.text

    def test_clean_markdown_content_with_code_excluded(self):
        """Test clean_markdown_content with code excluded."""
        content = """# Header
```python
def hello():
    print("Hello")
```
More content."""

        result = clean_markdown_content(content=content, include_code=False)
        assert "```python" not in result.text
        assert "def hello():" not in result.text
        assert "More content." in result.text


class TestCleanMarkdownFileEdgeCases:
    """Test edge cases for clean_markdown_file."""

    def test_clean_markdown_file_not_found(self):
        """Test clean_markdown_file with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found: nonexistent.md"):
            clean_markdown_file("nonexistent.md")

    def test_clean_markdown_file_empty_file(self):
        """Test clean_markdown_file with empty file."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            temp_file = f.name

        try:
            result = clean_markdown_file(temp_file)
            assert result.text == ""
        finally:
            os.unlink(temp_file)

    def test_clean_markdown_file_with_encoding_detection(self):
        """Test clean_markdown_file with encoding detection."""
        import os
        import tempfile

        # Create file with UTF-8 content
        content = "Hello World\n# Header\nSome content"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_file = f.name

        try:
            result = clean_markdown_file(temp_file)
            assert "Hello World" in result.text
            assert "Header" in result.text
        finally:
            os.unlink(temp_file)

    def test_clean_markdown_file_with_latin1_encoding(self):
        """Test clean_markdown_file with Latin-1 encoding."""
        import os
        import tempfile

        # Create file with Latin-1 content
        content = "Hello World\n# Header\nSome content"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="latin-1"
        ) as f:
            f.write(content)
            temp_file = f.name

        try:
            result = clean_markdown_file(temp_file)
            assert "Hello World" in result.text
            assert "Header" in result.text
        finally:
            os.unlink(temp_file)


class TestCleanMarkdownStringEdgeCases:
    """Test edge cases for clean_markdown_string."""

    def test_clean_markdown_string_empty(self):
        """Test clean_markdown_string with empty string."""
        result = clean_markdown_string("")
        assert result.text == ""

    def test_clean_markdown_string_whitespace_only(self):
        """Test clean_markdown_string with whitespace only."""
        result = clean_markdown_string("   \n\t\n  ")
        assert result.text.strip() == ""

    def test_clean_markdown_string_with_code_included(self):
        """Test clean_markdown_string with code included."""
        content = """# Header
```python
def hello():
    print("Hello")
```
More content."""

        result = clean_markdown_string(content, include_code=True)
        assert "def hello():" in result.text
        assert "More content." in result.text

    def test_clean_markdown_string_with_code_excluded(self):
        """Test clean_markdown_string with code excluded."""
        content = """# Header
```python
def hello():
    print("Hello")
```
More content."""

        result = clean_markdown_string(content, include_code=False)
        assert "def hello():" not in result.text
        assert "More content." in result.text


class TestMarkdownCleaningEdgeCases:
    """Test edge cases for markdown cleaning functions."""

    def test_remove_html_comments_edge_cases(self):
        """Test remove_html_comments with edge cases."""
        # Test basic HTML comment
        content = "Text <!-- Comment --> more text"
        result = remove_html_comments(content)
        assert "<!-- Comment -->" not in result
        assert "Text" in result
        assert "more text" in result

        # Test HTML comment with spaces
        content = "Text <!-- Comment with spaces --> more text"
        result = remove_html_comments(content)
        assert "<!-- Comment with spaces -->" not in result

        # Test multiple HTML comments
        content = "Text <!-- Comment1 --> middle <!-- Comment2 --> end"
        result = remove_html_comments(content)
        assert "<!-- Comment1 -->" not in result
        assert "<!-- Comment2 -->" not in result
        assert "Text" in result
        assert "middle" in result
        assert "end" in result

        # Test HTML comment at start
        content = "<!-- Start comment -->Text"
        result = remove_html_comments(content)
        assert "<!-- Start comment -->" not in result
        assert "Text" in result

        # Test HTML comment at end
        content = "Text<!-- End comment -->"
        result = remove_html_comments(content)
        assert "<!-- End comment -->" not in result
        assert "Text" in result

    def test_remove_yaml_front_matter_edge_cases(self):
        """Test remove_yaml_front_matter with edge cases."""
        # Test basic YAML front matter
        content = """---
title: Test
author: John
---
# Content
This is content."""

        result = remove_yaml_front_matter(content)
        assert "title: Test" not in result
        assert "author: John" not in result
        assert "Content" in result

        # Test YAML front matter with complex values
        content = """---
title: "Complex Title"
description: |
  Multi-line
  description
tags: [tag1, tag2]
---
# Content"""

        result = remove_yaml_front_matter(content)
        assert "title:" not in result
        assert "description:" not in result
        assert "tags:" not in result
        assert "Content" in result

        # Test content without YAML front matter
        content = "# Content\nThis is content without YAML."
        result = remove_yaml_front_matter(content)
        assert result == content

    def test_remove_fenced_code_blocks_edge_cases(self):
        """Test remove_fenced_code_blocks with edge cases."""
        # Test basic fenced code block
        content = """# Header
```python
def hello():
    print("Hello")
```
More content."""

        result = remove_fenced_code_blocks(content)
        assert "```python" not in result
        assert "def hello():" not in result
        assert "More content." in result

        # Test code block without language
        content = """# Header
```
def hello():
    print("Hello")
```
More content."""

        result = remove_fenced_code_blocks(content)
        assert "```" not in result
        assert "def hello():" not in result
        assert "More content." in result

        # Test multiple code blocks
        content = """# Header
```python
def hello():
    print("Hello")
```
Middle text
```javascript
console.log("World");
```
End text."""

        result = remove_fenced_code_blocks(content)
        assert "```python" not in result
        assert "```javascript" not in result
        assert "def hello():" not in result
        assert "console.log" not in result
        assert "Middle text" in result
        assert "End text." in result

    def test_remove_indented_code_blocks_edge_cases(self):
        """Test remove_indented_code_blocks with edge cases."""
        # Test basic indented code block
        content = """# Header
    def hello():
        print("Hello")

More content."""

        result = remove_indented_code_blocks(content)
        assert "def hello():" not in result
        assert "print(" not in result
        assert "More content." in result

        # Test mixed indentation
        content = """# Header
    def hello():
        print("Hello")
            nested_indent()

More content."""

        result = remove_indented_code_blocks(content)
        assert "def hello():" not in result
        assert "print(" not in result
        assert "nested_indent()" not in result
        assert "More content." in result

        # Test content without indented blocks
        content = "# Header\nThis is regular content.\nNo indentation here."
        result = remove_indented_code_blocks(content)
        assert result == content

    def test_markdown_to_plain_text_edge_cases(self):
        """Test markdown_to_plain_text with edge cases."""
        # Test basic markdown conversion
        content = "# Header\n**Bold text** and *italic text*"
        result = markdown_to_plain_text(content)
        assert "Header" in result
        assert "Bold text" in result
        assert "italic text" in result
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result

        # Test content without markdown
        content = "Plain text without markdown"
        result = markdown_to_plain_text(content)
        assert result.strip() == content

        # Test content with links
        content = "Text with [link](http://example.com)"
        result = markdown_to_plain_text(content)
        assert "Text with" in result
        assert "link" in result
        assert "[link](http://example.com)" not in result

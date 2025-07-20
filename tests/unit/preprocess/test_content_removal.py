"""Unit tests for content removal functionality (HTML comments, YAML, code blocks)."""

from app.preprocess.markdown_cleaner import (
    remove_fenced_code_blocks,
    remove_html_comments,
    remove_indented_code_blocks,
    remove_yaml_front_matter,
)


class TestRemoveHtmlComments:
    """Test HTML comment removal functionality."""

    def test_remove_single_line_html_comment(self):
        """Test removing single-line HTML comment."""
        content = "Line 1\n<!-- This is a comment -->\nLine 3"
        result = remove_html_comments(content)
        expected = "Line 1\n\nLine 3"
        assert result == expected

    def test_remove_multiline_html_comment(self):
        """Test removing multi-line HTML comment."""
        content = "Line 1\n<!--\nThis is a\nmulti-line comment\n-->\nLine 6"
        result = remove_html_comments(content)
        expected = "Line 1\n\n\n\n\nLine 6"
        assert result == expected

    def test_preserve_content_without_comments(self):
        """Test that content without comments is preserved."""
        content = "Line 1\nLine 2\nLine 3"
        result = remove_html_comments(content)
        assert result == content

    def test_remove_multiple_comments(self):
        """Test removing multiple HTML comments."""
        content = "Line 1\n<!-- Comment 1 -->\nLine 3\n<!-- Comment 2 -->\nLine 5"
        result = remove_html_comments(content)
        expected = "Line 1\n\nLine 3\n\nLine 5"
        assert result == expected

    def test_remove_html_comment_with_newlines(self):
        """Test removing HTML comment with newlines."""
        content = "Line 1\n<!--\nComment\n-->\nLine 5"
        result = remove_html_comments(content)
        expected = "Line 1\n\n\n\nLine 5"
        assert result == expected


class TestRemoveYamlFrontMatter:
    """Test YAML front matter removal functionality."""

    def test_remove_yaml_front_matter(self):
        """Test removing YAML front matter."""
        content = """---
title: Test Document
author: John Doe
date: 2023-01-01
---

# Main Content
This is the actual content."""
        result = remove_yaml_front_matter(content)
        expected = "\n\n\n\n\n\n# Main Content\nThis is the actual content."
        assert result == expected

    def test_preserve_content_without_yaml(self):
        """Test that content without YAML front matter is preserved."""
        content = "# Main Content\nThis is the actual content."
        result = remove_yaml_front_matter(content)
        assert result == content

    def test_handle_incomplete_yaml(self):
        """Test handling incomplete YAML front matter."""
        content = """---
title: Test Document
author: John Doe

# Main Content
This is the actual content."""
        result = remove_yaml_front_matter(content)
        expected = "---\ntitle: Test Document\nauthor: John Doe\n\n# Main Content\nThis is the actual content."
        assert result == expected

    def test_handle_yaml_with_content_after(self):
        """Test handling YAML with content after closing dashes."""
        content = """---
title: Test Document
---

# Main Content
This is the actual content."""
        result = remove_yaml_front_matter(content)
        expected = "\n\n\n\n# Main Content\nThis is the actual content."
        assert result == expected

    def test_handle_empty_content(self):
        """Test handling empty content."""
        content = ""
        result = remove_yaml_front_matter(content)
        assert result == ""

    def test_handle_content_without_dashes(self):
        """Test handling content without YAML dashes."""
        content = "title: Test Document\nauthor: John Doe\n\n# Main Content"
        result = remove_yaml_front_matter(content)
        assert result == content


class TestRemoveFencedCodeBlocks:
    """Test fenced code block removal functionality."""

    def test_remove_fenced_code_block(self):
        """Test removing fenced code block."""
        content = """# Header
```python
def hello():
    print("Hello, world!")
```
# Another Header"""
        result = remove_fenced_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n\n\n# Another Header"
        assert result == expected

    def test_remove_fenced_code_block_with_language(self):
        """Test removing fenced code block with language specification."""
        content = """# Header
```python
def hello():
    print("Hello, world!")
```
# Another Header"""
        result = remove_fenced_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n\n\n# Another Header"
        assert result == expected

    def test_preserve_code_blocks_when_include_code_true(self):
        """Test preserving code blocks when include_code is True."""
        content = """# Header
```python
def hello():
    print("Hello, world!")
```
# Another Header"""
        result = remove_fenced_code_blocks(content, include_code=True)
        assert result == content

    def test_handle_multiple_code_blocks(self):
        """Test handling multiple code blocks."""
        content = """# Header
```python
def hello():
    print("Hello, world!")
```
Text between blocks
```javascript
console.log("Hello");
```
# Another Header"""
        result = remove_fenced_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n\n\nText between blocks\n\n\n\n# Another Header"
        assert result == expected

    def test_handle_unclosed_code_block(self):
        """Test handling unclosed code block."""
        content = """# Header
```python
def hello():
    print("Hello, world!")
# Another Header"""
        result = remove_fenced_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n\n"
        assert result == expected


class TestRemoveIndentedCodeBlocks:
    """Test indented code block removal functionality."""

    def test_remove_indented_code_block(self):
        """Test removing indented code block."""
        content = """# Header
    def hello():
        print("Hello, world!")
# Another Header"""
        result = remove_indented_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n# Another Header"
        assert result == expected

    def test_remove_indented_code_block_with_tabs(self):
        """Test removing indented code block with tabs."""
        content = """# Header
\tdef hello():
\t\tprint("Hello, world!")
# Another Header"""
        result = remove_indented_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n# Another Header"
        assert result == expected

    def test_preserve_indented_comments(self):
        """Test preserving indented comments."""
        content = """# Header
    # This is a comment
    # Another comment
# Another Header"""
        result = remove_indented_code_blocks(content, include_code=False)
        expected = (
            "# Header\n    # This is a comment\n    # Another comment\n# Another Header"
        )
        assert result == expected

    def test_preserve_code_blocks_when_include_code_true(self):
        """Test preserving code blocks when include_code is True."""
        content = """# Header
    def hello():
        print("Hello, world!")
# Another Header"""
        result = remove_indented_code_blocks(content, include_code=True)
        assert result == content

    def test_handle_mixed_indentation(self):
        """Test handling mixed indentation."""
        content = """# Header
    def hello():
        print("Hello, world!")
        if True:
            print("True")
# Another Header"""
        result = remove_indented_code_blocks(content, include_code=False)
        expected = "# Header\n\n\n\n\n# Another Header"
        assert result == expected

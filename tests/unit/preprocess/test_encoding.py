"""Unit tests for encoding detection functionality."""

from unittest.mock import patch

from app.preprocess.markdown_cleaner import detect_encoding


class TestDetectEncoding:
    """Test encoding detection functionality."""

    def test_detect_utf8_encoding(self):
        """Test UTF-8 encoding detection."""
        content = "Hello, world! 🌍"
        file_bytes = content.encode("utf-8")
        encoding = detect_encoding(file_bytes)
        assert encoding == "utf-8"

    def test_detect_latin1_encoding(self):
        """Test Latin-1 encoding detection."""
        # Create bytes that can't be decoded as UTF-8
        file_bytes = b"\x80\x81\x82"
        encoding = detect_encoding(file_bytes)
        assert encoding == "latin-1"

    def test_fallback_to_latin1(self):
        """Test fallback to Latin-1 when detection fails."""
        # Create bytes that can't be decoded as UTF-8
        file_bytes = b"\x80\x81\x82"
        encoding = detect_encoding(file_bytes)
        assert encoding == "latin-1"

    @patch("app.preprocess.markdown_cleaner.chardet.detect")
    def test_detect_encoding_with_chardet(self, mock_detect):
        """Test encoding detection using chardet."""
        # Mock chardet to return a valid encoding
        mock_detect.return_value = {"encoding": "iso-8859-1", "confidence": 0.8}

        # Use bytes that can't be decoded as UTF-8 to force chardet usage
        file_bytes = b"\x80\x81\x82"
        encoding = detect_encoding(file_bytes)
        assert encoding == "iso-8859-1"

    @patch("app.preprocess.markdown_cleaner.chardet.detect")
    def test_detect_encoding_chardet_low_confidence(self, mock_detect):
        """Test encoding detection with low confidence from chardet."""
        # Mock chardet to return low confidence
        mock_detect.return_value = {"encoding": "iso-8859-1", "confidence": 0.5}

        # Use bytes that can't be decoded as UTF-8 to force chardet usage
        file_bytes = b"\x80\x81\x82"
        encoding = detect_encoding(file_bytes)
        assert encoding == "latin-1"  # Should fallback to latin-1

    @patch("app.preprocess.markdown_cleaner.chardet.detect")
    def test_detect_encoding_chardet_decode_error(self, mock_detect):
        """Test encoding detection when chardet result fails to decode."""
        # Mock chardet to return an encoding that fails to decode
        mock_detect.return_value = {"encoding": "invalid-encoding", "confidence": 0.9}

        # Use bytes that can't be decoded as UTF-8 to force chardet usage
        file_bytes = b"\x80\x81\x82"
        encoding = detect_encoding(file_bytes)
        assert encoding == "latin-1"  # Should fallback to latin-1

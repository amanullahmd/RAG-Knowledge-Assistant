"""Tests for document processing service"""

import pytest
from backend.app.services.document_processor import DocumentProcessor
from backend.app.core.exceptions import DocumentProcessingError


class TestDocumentProcessor:
    """Test suite for DocumentProcessor"""

    def test_init_defaults(self):
        """Test default initialization uses settings"""
        proc = DocumentProcessor()
        assert proc.chunk_size > 0
        assert proc.chunk_overlap > 0
        assert proc.chunk_overlap < proc.chunk_size

    def test_init_custom(self):
        """Test custom initialization"""
        proc = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        assert proc.chunk_size == 100
        assert proc.chunk_overlap == 20

    def test_supported_formats(self):
        """Test supported file formats"""
        proc = DocumentProcessor()
        assert ".pdf" in proc.SUPPORTED_FORMATS
        assert ".docx" in proc.SUPPORTED_FORMATS
        assert ".txt" in proc.SUPPORTED_FORMATS
        assert ".md" in proc.SUPPORTED_FORMATS

    def test_unsupported_format_raises(self, doc_processor):
        """Test that unsupported formats raise error"""
        with pytest.raises(DocumentProcessingError):
            doc_processor.process_file("test.xyz", b"content")

    def test_unsupported_format_error_message(self, doc_processor):
        """Test error message for unsupported format"""
        with pytest.raises(DocumentProcessingError) as exc_info:
            doc_processor.process_file("test.csv", b"content")
        assert ".csv" in str(exc_info.value.detail)

    def test_process_txt_file(self, doc_processor, sample_text_content):
        """Test processing a TXT file"""
        text, chunks, metadata = doc_processor.process_file(
            "test.txt", sample_text_content
        )
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert isinstance(metadata, list)
        assert len(metadata) == len(chunks)

    def test_process_md_file(self, doc_processor):
        """Test processing a Markdown file"""
        md_content = b"# Title\n\nSome **bold** content.\n\n## Section\n\nMore text here."
        text, chunks, metadata = doc_processor.process_file("readme.md", md_content)
        assert "Title" in text
        assert len(chunks) > 0
        assert all(m.get("source") == "readme.md" for m in metadata)

    def test_process_txt_metadata_has_source(self, doc_processor, sample_text_content):
        """Test that metadata includes source filename"""
        _, _, metadata = doc_processor.process_file("report.txt", sample_text_content)
        for m in metadata:
            assert m["source"] == "report.txt"

    def test_process_empty_txt(self, doc_processor):
        """Test processing an empty text file"""
        text, chunks, metadata = doc_processor.process_file("empty.txt", b"")
        assert len(chunks) == 1
        assert len(metadata) == 1

    def test_process_unicode_txt(self, doc_processor):
        """Test processing UTF-8 content"""
        content = "Héllo wörld café résumé naïve".encode("utf-8")
        text, chunks, metadata = doc_processor.process_file("unicode.txt", content)
        assert "Héllo" in text
        assert len(chunks) > 0

    def test_process_latin1_fallback(self, doc_processor):
        """Test processing falls back to Latin-1 for non-UTF-8"""
        content = "Héllo wörld".encode("latin-1")
        text, chunks, metadata = doc_processor.process_file("latin.txt", content)
        assert len(chunks) > 0

    def test_chunk_text_basic(self, small_doc_processor):
        """Test basic text chunking"""
        text = " ".join(["word"] * 100)
        chunks = small_doc_processor._chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunk_text_empty(self, doc_processor):
        """Test chunking empty text"""
        chunks = doc_processor._chunk_text("")
        assert chunks == []

    def test_chunk_text_small(self, doc_processor):
        """Test chunking text smaller than chunk size"""
        text = "Short text"
        chunks = doc_processor._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_overlap(self, small_doc_processor):
        """Test that chunks have overlapping content"""
        text = " ".join(f"word{i}" for i in range(50))
        chunks = small_doc_processor._chunk_text(text)
        if len(chunks) >= 2:
            # Check that there's some word overlap between consecutive chunks
            words_first = set(chunks[0].split())
            words_second = set(chunks[1].split())
            overlap = words_first & words_second
            assert len(overlap) > 0, "Chunks should have overlapping words"

    def test_chunks_cover_all_content(self, small_doc_processor):
        """Test that all words in the text are covered by at least one chunk"""
        text = " ".join(f"word{i}" for i in range(30))
        chunks = small_doc_processor._chunk_text(text)
        all_words = set(text.split())
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(chunk.split())
        assert all_words == chunk_words

    def test_process_file_returns_tuple(self, doc_processor, sample_text_content):
        """Test that process_file returns a 3-tuple"""
        result = doc_processor.process_file("test.txt", sample_text_content)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestDocumentProcessorEdgeCases:
    """Edge case tests"""

    def test_very_large_text(self, doc_processor):
        """Test processing a very large text"""
        content = ("Lorem ipsum dolor sit amet. " * 1000).encode("utf-8")
        text, chunks, metadata = doc_processor.process_file("large.txt", content)
        assert len(chunks) > 1
        assert len(metadata) == len(chunks)

    def test_filename_with_spaces(self, doc_processor, sample_text_content):
        """Test filename with spaces"""
        _, _, metadata = doc_processor.process_file(
            "my document file.txt", sample_text_content
        )
        assert metadata[0]["source"] == "my document file.txt"

    def test_filename_case_insensitive_extension(self, doc_processor, sample_text_content):
        """Test that file extension is case-insensitive"""
        text, chunks, metadata = doc_processor.process_file(
            "TEST.TXT", sample_text_content
        )
        assert len(chunks) > 0

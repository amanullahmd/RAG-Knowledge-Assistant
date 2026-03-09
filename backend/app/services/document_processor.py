"""Document processing service"""

import os
import io
from typing import List, Tuple
from pathlib import Path
import mimetypes

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from backend.app.core.config import settings
from backend.app.core.exceptions import DocumentProcessingError

class DocumentProcessor:
    """Handles multi-format document parsing and chunking"""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.md'}
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, filename: str, file_content: bytes) -> Tuple[str, List[str], List[dict]]:
        """
        Process uploaded file and return text, chunks, and metadata
        
        Args:
            filename: Name of the file
            file_content: Binary content of the file
            
        Returns:
            Tuple of (text, chunks, metadata_list)
        """
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise DocumentProcessingError(
                f"Unsupported file type: {file_ext}. Supported: {self.SUPPORTED_FORMATS}"
            )
        
        if file_ext == '.pdf':
            return self._process_pdf(filename, file_content)
        elif file_ext == '.docx':
            return self._process_docx(filename, file_content)
        elif file_ext in {'.txt', '.md'}:
            return self._process_text(filename, file_content)
    
    def _process_pdf(self, filename: str, file_content: bytes) -> Tuple[str, List[str], List[dict]]:
        """Process PDF file"""
        if not pymupdf4llm:
            raise DocumentProcessingError("PDF processing requires pymupdf4llm")
        
        try:
            import pymupdf
            pdf_bytes = io.BytesIO(file_content)
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            
            text = ""
            page_map = {}
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
                page_map[page_num + 1] = page.get_text()
            
            doc.close()
            chunks = self._chunk_text(text)
            metadata = [
                {"page": (len(chunks) // len(page_map)) * (i // len(chunks) + 1), "source": filename}
                for i in range(len(chunks))
            ]
            
            return text, chunks, metadata
        except Exception as e:
            raise DocumentProcessingError(f"PDF processing failed: {str(e)}")
    
    def _process_docx(self, filename: str, file_content: bytes) -> Tuple[str, List[str], List[dict]]:
        """Process DOCX file"""
        if not DocxDocument:
            raise DocumentProcessingError("DOCX processing requires python-docx")
        
        try:
            docx_bytes = io.BytesIO(file_content)
            doc = DocxDocument(docx_bytes)
            
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + "\n"
            
            chunks = self._chunk_text(text)
            metadata = [{"source": filename} for _ in chunks]
            
            return text, chunks, metadata
        except Exception as e:
            raise DocumentProcessingError(f"DOCX processing failed: {str(e)}")
    
    def _process_text(self, filename: str, file_content: bytes) -> Tuple[str, List[str], List[dict]]:
        """Process TXT or MD file"""
        try:
            text = file_content.decode('utf-8')
            chunks = self._chunk_text(text)
            metadata = [{"source": filename} for _ in chunks]
            
            return text, chunks, metadata
        except Exception as e:
            raise DocumentProcessingError(f"Text processing failed: {str(e)}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        Simple implementation - can be enhanced with heading-aware splitting
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        words_per_chunk = max(self.chunk_size // 4, 1)  # rough estimate
        stride = max(words_per_chunk - (self.chunk_overlap // 4), 1)
        
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i:i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]

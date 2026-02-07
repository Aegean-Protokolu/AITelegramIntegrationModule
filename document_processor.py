import os
import re
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np


class DocumentProcessor:
    """
    Document processor for PDF and DOCX files.
    
    Features:
    - Extract text from PDF (PyMuPDF/fitz) and DOCX (python-docx)
    - Intelligent chunking with overlap
    - Embedding generation
    - Metadata extraction
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray], chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            embed_fn: Function to generate embeddings (from lm.py)
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.embed_fn = embed_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # --------------------------
    # PDF Processing
    # --------------------------

    def process_pdf(self, file_path: str) -> Tuple[List[Dict], int]:
        """
        Extract text from PDF and chunk it.
        
        Returns:
            (chunks, page_count)
            chunks = [{'text': str, 'embedding': np.ndarray, 'page_number': int}, ...]
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF (fitz) not installed. Run: pip install PyMuPDF")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"Could not open PDF: {e}")

        page_count = len(doc)

        # Extract text from each page
        pages = []
        for page_num in range(page_count):
            try:
                page = doc[page_num]
                text = page.get_text("text")
                pages.append({
                    'page_number': page_num + 1,  # 1-indexed
                    'text': text
                })
            except Exception as e:
                print(f"⚠️ Warning: Skipping page {page_num + 1} in {os.path.basename(file_path)} due to error: {e}")
                continue

        doc.close()
        
        if not pages:
            raise ValueError("No text extracted from PDF (file might be empty, corrupted, or contain only images).")

        # Chunk the pages
        chunks = self._chunk_pages(pages)

        # Generate embeddings
        for chunk in chunks:
            chunk['embedding'] = self.embed_fn(chunk['text'])

        return chunks, page_count

    # --------------------------
    # DOCX Processing
    # --------------------------

    def process_docx(self, file_path: str) -> Tuple[List[Dict], None]:
        """
        Extract text from DOCX and chunk it.
        
        Returns:
            (chunks, None)  # DOCX doesn't have fixed page numbers
            chunks = [{'text': str, 'embedding': np.ndarray}, ...]
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        doc = Document(file_path)

        # Extract all paragraphs
        full_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        # Chunk the text
        chunks = self._chunk_text(full_text)

        # Generate embeddings
        for chunk in chunks:
            chunk['embedding'] = self.embed_fn(chunk['text'])

        return chunks, None  # No page count for DOCX

    # --------------------------
    # Chunking Strategy
    # --------------------------

    def _chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Chunk text across multiple pages with overlap.
        
        Preserves page number information.
        """
        chunks = []
        current_chunk = ""
        current_page_start = pages[0]['page_number'] if pages else 1

        for page in pages:
            page_text = page['text']

            # Add page text to current chunk
            current_chunk += page_text + "\n"

            # If chunk exceeds size, split it
            while len(current_chunk) >= self.chunk_size:
                # Extract chunk
                chunk_text = current_chunk[:self.chunk_size]
                
                # Try to break at sentence boundary
                chunk_text = self._break_at_sentence(chunk_text)

                chunks.append({
                    'text': chunk_text.strip(),
                    'page_number': current_page_start
                })

                # Overlap for next chunk
                current_chunk = current_chunk[len(chunk_text) - self.chunk_overlap:]
                current_page_start = page['page_number']

        # Add remaining text as final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page_number': current_page_start
            })

        return chunks

    def _chunk_text(self, text: str) -> List[Dict]:
        """
        Chunk plain text with overlap (for DOCX).
        """
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Extract chunk
            chunk_text = text[current_pos:current_pos + self.chunk_size]

            # Try to break at sentence boundary
            if current_pos + self.chunk_size < len(text):
                chunk_text = self._break_at_sentence(chunk_text)

            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip()
                })

            # Move position with overlap
            current_pos += len(chunk_text) - self.chunk_overlap

        return chunks

    def _break_at_sentence(self, text: str) -> str:
        """
        Try to break text at sentence boundary (., !, ?, \n).
        """
        # Find last sentence-ending punctuation.
        # Look for [.!?] followed by whitespace and an Uppercase letter, OR a newline.
        matches = list(re.finditer(r'(?<=[.!?])\s+(?=[A-Z])|\n', text))
        
        if matches:
            # Get last match position
            last_match = matches[-1]
            return text[:last_match.end()]
        
        # Fallback: break at last space
        last_space = text.rfind(' ')
        if last_space > 0:
            return text[:last_space]
        
        # Fallback: return as is
        return text

    # --------------------------
    # Utilities
    # --------------------------

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text (remove excessive whitespace, etc.).
        """
        # Fix hyphenated words at line ends (e.g. "process-\ning" -> "processing")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Un-wrap lines: Replace single newlines with space, keep double newlines for paragraphs.
        # Look for newline NOT preceded by newline AND NOT followed by newline.
        # This joins sentences split by PDF hard wraps.
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def get_file_type(self, filename: str) -> Optional[str]:
        """
        Determine file type from extension.
        
        Returns:
            'pdf', 'docx', or None
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.docx', '.doc']:
            return 'docx'
        return None

    def process_document(self, file_path: str) -> Tuple[List[Dict], Optional[int], str]:
        """
        Auto-detect file type and process.
        
        Returns:
            (chunks, page_count, file_type)
        
        Raises:
            ValueError: If file type not supported
        """
        file_type = self.get_file_type(file_path)

        if file_type == 'pdf':
            chunks, page_count = self.process_pdf(file_path)
            return chunks, page_count, 'pdf'
        elif file_type == 'docx':
            chunks, _ = self.process_docx(file_path)
            return chunks, None, 'docx'
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

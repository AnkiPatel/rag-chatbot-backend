from typing import List, Dict, Optional
from pathlib import Path
import pypdf
from dataclasses import dataclass
from app.config import settings
from app.utils.logger import app_logger as logger


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    content: str
    metadata: Dict[str, any]
    chunk_id: str


class PDFProcessor:
    """Service for processing PDF documents and extracting text."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        logger.info(f"Initialized PDFProcessor with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def load_pdf(self, file_path: str) -> Dict[str, any]:
        """
        Load and extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing full text and metadata
        """
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            logger.info(f"Loading PDF: {pdf_path.name}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_texts.append({
                        'page_number': page_num + 1,
                        'text': page_text
                    })
                    full_text += f"\n{page_text}\n"
                
                metadata = {
                    'filename': pdf_path.name,
                    'filepath': str(pdf_path),
                    'num_pages': len(pdf_reader.pages),
                    'total_chars': len(full_text)
                }
                
                logger.info(f"Successfully loaded {pdf_path.name}: {metadata['num_pages']} pages, {metadata['total_chars']} chars")
                
                return {
                    'full_text': full_text.strip(),
                    'page_texts': page_texts,
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, any]) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Full text to chunk
            metadata: Document metadata to attach to chunks
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_num = 0
        
        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for last sentence ending in the chunk
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # Only break if we don't lose too much
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunk_metadata = {
                **metadata,
                'chunk_number': chunk_num,
                'start_char': start,
                'end_char': end
            }
            
            chunk = DocumentChunk(
                content=chunk_text.strip(),
                metadata=chunk_metadata,
                chunk_id=f"{metadata.get('filename', 'unknown')}_{chunk_num}"
            )
            
            chunks.append(chunk)
            chunk_num += 1
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """
        Load PDF and split into chunks in one operation.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        doc_data = self.load_pdf(file_path)
        chunks = self.chunk_text(doc_data['full_text'], doc_data['metadata'])
        return chunks
    
    def process_directory(self, directory: Optional[str] = None) -> List[DocumentChunk]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Directory path (defaults to settings.pdf_directory)
            
        Returns:
            List of all DocumentChunk objects from all PDFs
        """
        dir_path = Path(directory or settings.pdf_directory)
        
        if not dir_path.exists():
            logger.warning(f"PDF directory does not exist: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            return []
        
        pdf_files = list(dir_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")
        
        all_chunks = []
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(pdf_files)} PDFs into {len(all_chunks)} total chunks")
        return all_chunks

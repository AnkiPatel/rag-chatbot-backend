"""
Script to initialize the knowledge base by processing and indexing PDF documents.
Run this script after adding PDFs to the data/pdfs directory.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.utils.logger import app_logger as logger


def main():
    """Initialize knowledge base by processing and indexing PDFs."""
    
    logger.info("=" * 60)
    logger.info("Knowledge Base Initialization")
    logger.info("=" * 60)
    
    # Initialize services
    logger.info("Initializing services...")
    processor = PDFProcessor()
    store = VectorStore()
    
    # Get current stats
    initial_stats = store.get_stats()
    logger.info(f"Current state: {initial_stats['total_chunks']} chunks from {initial_stats['unique_files']} files")
    
    # Process PDFs
    logger.info("Processing PDF documents...")
    chunks = processor.process_directory()
    
    if not chunks:
        logger.warning("No PDF files found or processed. Please add PDF files to data/pdfs/")
        return
    
    # Index documents
    logger.info(f"Indexing {len(chunks)} chunks...")
    store.add_documents(chunks)
    
    # Get final stats
    final_stats = store.get_stats()
    
    logger.info("=" * 60)
    logger.info("Initialization Complete!")
    logger.info(f"Total chunks indexed: {final_stats['total_chunks']}")
    logger.info(f"Unique files: {final_stats['unique_files']}")
    logger.info(f"Embedding model: {final_stats['embedding_model']}")
    logger.info(f"Embedding dimension: {final_stats['embedding_dimension']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from typing import List
from app.api.models.schemas import UploadResponse, DocumentListResponse
from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.config import settings
from app.utils.logger import app_logger as logger

router = APIRouter()

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStore()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to the knowledge base.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info(f"Uploading PDF: {file.filename}")
        
        # Save file
        pdf_dir = Path(settings.pdf_directory)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = pdf_dir / file.filename
        
        # Write uploaded file
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved PDF to: {file_path}")
        
        # Process and index the PDF
        chunks = pdf_processor.process_pdf(str(file_path))
        vector_store.add_documents(chunks)
        
        # Get updated stats
        stats = vector_store.get_stats()
        
        return UploadResponse(
            message=f"Successfully uploaded and indexed {file.filename}",
            filename=file.filename,
            chunks_created=len(chunks),
            total_documents=stats['total_chunks']
        )
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents in the knowledge base.
    """
    try:
        pdf_dir = Path(settings.pdf_directory)
        
        if not pdf_dir.exists():
            return DocumentListResponse(documents=[], total_count=0)
        
        pdf_files = [f.name for f in pdf_dir.glob("*.pdf")]
        
        return DocumentListResponse(
            documents=sorted(pdf_files),
            total_count=len(pdf_files)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document from the knowledge base.
    """
    try:
        logger.info(f"Deleting document: {filename}")
        
        # Delete from vector store
        vector_store.delete_by_filename(filename)
        
        # Delete file
        pdf_path = Path(settings.pdf_directory) / filename
        if pdf_path.exists():
            pdf_path.unlink()
            logger.info(f"Deleted file: {pdf_path}")
        
        return {"message": f"Successfully deleted {filename}"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.post("/reindex")
async def reindex_all():
    """
    Reindex all PDF documents in the knowledge base.
    """
    try:
        logger.info("Starting reindexing of all documents")
        
        # Clear existing collection
        vector_store.clear_collection()
        
        # Process all PDFs
        chunks = pdf_processor.process_directory()
        
        # Index all chunks
        if chunks:
            vector_store.add_documents(chunks)
        
        stats = vector_store.get_stats()
        
        return {
            "message": "Reindexing completed successfully",
            "total_chunks": stats['total_chunks'],
            "unique_files": stats['unique_files']
        }
        
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during reindexing: {str(e)}")

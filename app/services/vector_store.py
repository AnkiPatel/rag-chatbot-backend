from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from app.config import settings
from app.core.embeddings import EmbeddingService
from app.services.pdf_processor import DocumentChunk
from app.utils.logger import app_logger as logger


class VectorStore:
    """Service for managing vector database operations using ChromaDB."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.db_path = Path(settings.vector_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {self.db_path}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"description": "Product knowledge base for RAG chatbot"}
        )
        
        logger.info(f"Collection '{settings.collection_name}' initialized with {self.collection.count()} documents")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to index
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return
        
        logger.info(f"Adding {len(chunks)} document chunks to vector store")
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings_batch(documents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks. Total documents: {self.collection.count()}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document content, metadata, and scores
        """
        logger.info(f"Searching for: '{query[:100]}...' (top {k} results)")
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count())
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    'id': results['ids'][0][i] if results['ids'] else None
                })
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: ID of the document to delete
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            raise
    
    def delete_by_filename(self, filename: str) -> None:
        """
        Delete all chunks from a specific file.
        
        Args:
            filename: Name of the file whose chunks to delete
        """
        try:
            # Query all documents from this file
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks from file: {filename}")
            else:
                logger.info(f"No chunks found for file: {filename}")
                
        except Exception as e:
            logger.error(f"Error deleting chunks for {filename}: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get unique filenames
        if count > 0:
            all_docs = self.collection.get()
            unique_files = set()
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    if 'filename' in metadata:
                        unique_files.add(metadata['filename'])
            num_files = len(unique_files)
        else:
            num_files = 0
        
        stats = {
            'total_chunks': count,
            'unique_files': num_files,
            'collection_name': settings.collection_name,
            'embedding_model': self.embedding_service.get_model_name(),
            'embedding_dimension': self.embedding_service.get_dimension()
        }
        
        return stats
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        logger.warning("Clearing all documents from collection")
        self.client.delete_collection(name=settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"description": "Product knowledge base for RAG chatbot"}
        )
        logger.info("Collection cleared and recreated")

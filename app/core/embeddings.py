from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logger import app_logger as logger


class EmbeddingService:
    """Service for generating text embeddings using sentence transformers."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Embedding model loaded successfully. Dimension: {self.get_dimension()}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.get_dimension()
        
        embedding = self._model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_tensor=False
        )
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return [emb.tolist() for emb in embeddings]
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get the name of the loaded model."""
        return settings.embedding_model

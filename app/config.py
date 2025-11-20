from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""
    
    # LLM Configuration
    llm_provider: str = "openai"
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000
    
    # Vector Database Configuration
    vector_db_type: str = "chromadb"
    vector_db_path: str = "./data/vector_db"
    collection_name: str = "product_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Web Search Configuration
    search_api_provider: str = "tavily"
    tavily_api_key: str
    max_search_results: int = 5
    
    # PDF Processing
    pdf_directory: str = "./data/pdfs"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "RAG Chatbot API"
    api_version: str = "1.0.0"
    allowed_origins: str = "*"
    
    # Application
    log_level: str = "INFO"
    environment: str = "development"
    
    # Optional API Key for endpoint security
    api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

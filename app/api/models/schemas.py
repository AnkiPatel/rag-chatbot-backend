from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., description="User's question", min_length=1)
    use_search: bool = Field(True, description="Whether to use web search if needed")
    num_results: int = Field(5, description="Number of knowledge base results to retrieve", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I configure the admin panel?",
                "use_search": True,
                "num_results": 5
            }
        }


class Source(BaseModel):
    """Source information for a response."""
    type: str = Field(..., description="Type of source: 'knowledge_base' or 'web_search'")
    filename: Optional[str] = Field(None, description="Filename for knowledge base sources")
    page_number: Optional[str] = Field(None, description="Page number for knowledge base sources")
    chunk_number: Optional[str] = Field(None, description="Chunk number for knowledge base sources")
    title: Optional[str] = Field(None, description="Title for web search sources")
    url: Optional[str] = Field(None, description="URL for web search sources")
    relevance_score: float = Field(..., description="Relevance score of this source")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="Generated answer to the user's question")
    sources: List[Source] = Field([], description="Sources used to generate the answer")
    confidence: float = Field(..., description="Confidence score of the answer (0-1)")
    used_web_search: bool = Field(..., description="Whether web search was used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To configure the admin panel, navigate to Settings > Admin Configuration...",
                "sources": [
                    {
                        "type": "knowledge_base",
                        "filename": "admin_guide.pdf",
                        "page_number": "12",
                        "chunk_number": "5",
                        "relevance_score": 0.92
                    }
                ],
                "confidence": 0.92,
                "used_web_search": False
            }
        }


class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    filename: str
    chunks_created: int
    total_documents: int


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    filename: str
    num_chunks: int


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[str]
    total_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str


class DetailedHealthResponse(BaseModel):
    """Response model for detailed health check."""
    status: str
    timestamp: str
    llm_status: str
    vector_db_status: str
    search_status: str
    documents_indexed: int
    embedding_model: str
    llm_model: str

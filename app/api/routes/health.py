from fastapi import APIRouter
from datetime import datetime
from app.api.models.schemas import HealthResponse, DetailedHealthResponse
from app.core.rag_pipeline import RAGPipeline
from app.utils.logger import app_logger as logger

router = APIRouter()

# Initialize pipeline for health checks
try:
    pipeline = RAGPipeline()
except Exception as e:
    logger.error(f"Failed to initialize pipeline in health routes: {str(e)}")
    pipeline = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with component status."""
    
    llm_status = "unknown"
    vector_db_status = "unknown"
    search_status = "unknown"
    documents_indexed = 0
    embedding_model = "unknown"
    llm_model = "unknown"
    
    if pipeline:
        try:
            # Check vector DB
            stats = pipeline.vector_store.get_stats()
            vector_db_status = "ok"
            documents_indexed = stats.get('total_chunks', 0)
            embedding_model = stats.get('embedding_model', 'unknown')
            
            # LLM status
            llm_status = "ok"
            llm_model = pipeline.llm_client.model
            
            # Search status
            search_status = "ok"
            
        except Exception as e:
            logger.error(f"Error in detailed health check: {str(e)}")
    
    overall_status = "healthy" if all(
        s == "ok" for s in [llm_status, vector_db_status, search_status]
    ) else "degraded"
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        llm_status=llm_status,
        vector_db_status=vector_db_status,
        search_status=search_status,
        documents_indexed=documents_indexed,
        embedding_model=embedding_model,
        llm_model=llm_model
    )

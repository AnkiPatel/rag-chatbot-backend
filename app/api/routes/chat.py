from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.api.models.schemas import ChatRequest, ChatResponse, Source
from app.core.rag_pipeline import RAGPipeline
from app.utils.logger import app_logger as logger

router = APIRouter()

# Initialize RAG pipeline
pipeline = RAGPipeline()


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Process a user query and return an answer with sources.
    """
    try:
        logger.info(f"Received chat query: {request.query[:100]}...")
        
        # Process query through RAG pipeline
        response = await pipeline.query(
            user_query=request.query,
            use_search=request.use_search,
            num_results=request.num_results
        )
        
        # Convert sources to Pydantic models
        sources = [Source(**source) for source in response.sources]
        
        return ChatResponse(
            answer=response.answer,
            sources=sources,
            confidence=response.confidence,
            used_web_search=response.used_web_search
        )
        
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/stream")
async def chat_query_stream(request: ChatRequest):
    """
    Process a user query and return a streaming response.
    """
    try:
        logger.info(f"Received streaming chat query: {request.query[:100]}...")
        
        async def generate():
            async for chunk in pipeline.query_stream(
                user_query=request.query,
                use_search=request.use_search,
                num_results=request.num_results
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error processing streaming query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

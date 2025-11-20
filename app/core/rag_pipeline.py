from typing import List, Dict, Optional
from dataclasses import dataclass
from app.core.llm_client import LLMClient
from app.services.vector_store import VectorStore
from app.services.web_search import WebSearchService
from app.utils.logger import app_logger as logger


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    sources: List[Dict]
    confidence: float
    used_web_search: bool


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval and generation."""
    
    def __init__(self):
        logger.info("Initializing RAG Pipeline")
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
        self.web_search = WebSearchService()
        logger.info("RAG Pipeline initialized successfully")
    
    async def query(
        self,
        user_query: str,
        use_search: bool = True,
        num_results: int = 5,
        search_confidence_threshold: float = 0.7
    ) -> RAGResponse:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            user_query: The user's question
            use_search: Whether to use web search if knowledge base confidence is low
            num_results: Number of documents to retrieve from knowledge base
            search_confidence_threshold: Threshold for triggering web search
            
        Returns:
            RAGResponse object with answer and metadata
        """
        logger.info(f"Processing query: '{user_query[:100]}...'")
        
        # Step 1: Retrieve from knowledge base
        kb_results = self.vector_store.search(user_query, k=num_results)
        
        # Step 2: Determine if web search is needed
        used_search = False
        search_context = None
        search_sources = []
        
        if use_search and self.web_search.should_use_search(kb_results, search_confidence_threshold):
            logger.info("Knowledge base confidence low, performing web search")
            search_results = self.web_search.search(user_query)
            
            if search_results:
                used_search = True
                search_context = self.web_search.format_search_results_for_context(search_results)
                search_sources = [result.to_dict() for result in search_results]
        
        # Step 3: Generate response using LLM
        answer = self.llm_client.generate_response(
            query=user_query,
            knowledge_base_context=kb_results,
            search_results_context=search_context
        )
        
        # Step 4: Calculate confidence and format sources
        confidence = self._calculate_confidence(kb_results)
        sources = self._format_sources(kb_results, search_sources)
        
        response = RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            used_web_search=used_search
        )
        
        logger.info(f"Query processed successfully. Confidence: {confidence:.2f}, Used search: {used_search}")
        return response
    
    async def query_stream(
        self,
        user_query: str,
        use_search: bool = True,
        num_results: int = 5,
        search_confidence_threshold: float = 0.7
    ):
        """
        Process a user query with streaming response.
        
        Args:
            user_query: The user's question
            use_search: Whether to use web search
            num_results: Number of documents to retrieve
            search_confidence_threshold: Threshold for web search
            
        Yields:
            Chunks of the generated response
        """
        logger.info(f"Processing streaming query: '{user_query[:100]}...'")
        
        # Retrieve from knowledge base
        kb_results = self.vector_store.search(user_query, k=num_results)
        
        # Check if web search needed
        search_context = None
        if use_search and self.web_search.should_use_search(kb_results, search_confidence_threshold):
            search_results = self.web_search.search(user_query)
            if search_results:
                search_context = self.web_search.format_search_results_for_context(search_results)
        
        # Stream response
        async for chunk in self.llm_client.generate_response_stream(
            query=user_query,
            knowledge_base_context=kb_results,
            search_results_context=search_context
        ):
            yield chunk
    
    def _calculate_confidence(self, kb_results: List[Dict]) -> float:
        """
        Calculate confidence score based on retrieval results.
        
        Args:
            kb_results: Results from vector store
            
        Returns:
            Confidence score between 0 and 1
        """
        if not kb_results:
            return 0.0
        
        # Use distance from top result (lower distance = higher confidence)
        top_distance = kb_results[0].get('distance', 2.0)
        
        # Convert distance to confidence (distance of 0 = confidence 1.0, distance of 2 = confidence 0.0)
        confidence = max(0.0, min(1.0, 1.0 - (top_distance / 2.0)))
        
        return round(confidence, 2)
    
    def _format_sources(
        self,
        kb_results: List[Dict],
        search_sources: List[Dict]
    ) -> List[Dict]:
        """
        Format sources for the response.
        
        Args:
            kb_results: Knowledge base results
            search_sources: Web search sources
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        # Add knowledge base sources
        for result in kb_results[:3]:  # Top 3 KB sources
            metadata = result.get('metadata', {})
            sources.append({
                'type': 'knowledge_base',
                'filename': metadata.get('filename', 'Unknown'),
                'page_number': metadata.get('page_number', 'N/A'),
                'chunk_number': metadata.get('chunk_number', 'N/A'),
                'relevance_score': round(1.0 - result.get('distance', 1.0) / 2.0, 2)
            })
        
        # Add web search sources
        for search_source in search_sources[:3]:  # Top 3 web sources
            sources.append({
                'type': 'web_search',
                'title': search_source.get('title', 'No title'),
                'url': search_source.get('url', ''),
                'relevance_score': search_source.get('score', 0.0)
            })
        
        return sources
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the RAG pipeline."""
        return {
            'vector_store': self.vector_store.get_stats(),
            'llm_model': self.llm_client.model,
            'search_provider': 'Tavily'
        }

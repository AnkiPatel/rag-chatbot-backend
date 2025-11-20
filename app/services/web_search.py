from typing import List, Dict, Optional
from tavily import TavilyClient
from app.config import settings
from app.utils.logger import app_logger as logger


class SearchResult:
    """Represents a web search result."""
    
    def __init__(self, title: str, url: str, content: str, score: float = 0.0):
        self.title = title
        self.url = url
        self.content = content
        self.score = score
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'url': self.url,
            'content': self.content,
            'score': self.score
        }


class WebSearchService:
    """Service for performing web searches using Tavily API."""
    
    def __init__(self):
        self.client = TavilyClient(api_key=settings.tavily_api_key)
        self.max_results = settings.max_search_results
        logger.info("WebSearchService initialized with Tavily API")
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Perform a web search for the given query.
        
        Args:
            query: Search query
            max_results: Maximum number of results (defaults to settings.max_search_results)
            
        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.max_results
        
        try:
            logger.info(f"Performing web search: '{query[:100]}...'")
            
            # Perform search using Tavily
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
                include_raw_content=False
            )
            
            # Process results
            search_results = []
            
            if 'results' in response:
                for result in response['results']:
                    search_result = SearchResult(
                        title=result.get('title', 'No title'),
                        url=result.get('url', ''),
                        content=result.get('content', ''),
                        score=result.get('score', 0.0)
                    )
                    search_results.append(search_result)
            
            logger.info(f"Found {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return []
    
    def should_use_search(self, knowledge_base_results: List[Dict], confidence_threshold: float = 0.7) -> bool:
        """
        Determine if web search should be used based on knowledge base results.
        
        Args:
            knowledge_base_results: Results from vector store search
            confidence_threshold: Minimum confidence to skip search
            
        Returns:
            True if web search should be performed
        """
        if not knowledge_base_results:
            logger.info("No knowledge base results, will use web search")
            return True
        
        # Check if top result has high confidence (low distance)
        top_result_distance = knowledge_base_results[0].get('distance', 1.0)
        
        # ChromaDB uses distance (lower is better), convert to confidence
        # Distance of 0 = perfect match, distance of 2 = completely different
        confidence = 1.0 - (top_result_distance / 2.0)
        
        use_search = confidence < confidence_threshold
        logger.info(f"Top result confidence: {confidence:.2f}, threshold: {confidence_threshold}, use_search: {use_search}")
        
        return use_search
    
    def format_search_results_for_context(self, results: List[SearchResult]) -> str:
        """
        Format search results into a context string for the LLM.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return ""
        
        context_parts = ["Web Search Results:\n"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result.title}")
            context_parts.append(f"   Source: {result.url}")
            context_parts.append(f"   {result.content}\n")
        
        return "\n".join(context_parts)

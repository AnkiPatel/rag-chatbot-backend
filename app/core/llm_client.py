from typing import List, Dict, Optional
from openai import OpenAI
from app.config import settings
from app.utils.logger import app_logger as logger


class LLMClient:
    """Client for interacting with OpenAI's language models."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
        logger.info(f"Initialized LLMClient with model: {self.model}")
    
    def create_prompt(
        self,
        query: str,
        knowledge_base_context: List[Dict],
        search_results_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Create a prompt for the LLM with context from knowledge base and web search.
        
        Args:
            query: User's question
            knowledge_base_context: Results from vector store
            search_results_context: Formatted web search results
            
        Returns:
            List of message dictionaries for the API
        """
        # Build context from knowledge base
        kb_context_parts = []
        if knowledge_base_context:
            kb_context_parts.append("Internal Knowledge Base:\n")
            for i, doc in enumerate(knowledge_base_context, 1):
                source = doc.get('metadata', {}).get('filename', 'Unknown')
                page = doc.get('metadata', {}).get('page_number', 'N/A')
                kb_context_parts.append(f"[Source {i}: {source}, Page {page}]")
                kb_context_parts.append(doc.get('content', ''))
                kb_context_parts.append("")  # blank line
        
        kb_context = "\n".join(kb_context_parts) if kb_context_parts else "No relevant internal documentation found."
        
        # System message
        system_message = """You are a helpful AI assistant for a product support chatbot. Your role is to answer user questions accurately using the provided context from product documentation and web search results.

Guidelines:
- Always prioritize information from the Internal Knowledge Base (product documentation)
- Use web search results to supplement or provide additional context when needed
- If the answer is not in the provided context, clearly state that you don't have that information
- Be concise but comprehensive in your answers
- If referencing specific sources, mention them
- Maintain a professional and friendly tone"""
        
        # User message with context
        user_content_parts = [
            kb_context,
            ""
        ]
        
        if search_results_context:
            user_content_parts.append(search_results_context)
            user_content_parts.append("")
        
        user_content_parts.append(f"User Question: {query}")
        user_content_parts.append("\nPlease provide a helpful answer based on the context above:")
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n".join(user_content_parts)}
        ]
        
        return messages
    
    def generate_response(
        self,
        query: str,
        knowledge_base_context: List[Dict],
        search_results_context: Optional[str] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            query: User's question
            knowledge_base_context: Results from vector store
            search_results_context: Formatted web search results
            
        Returns:
            Generated response text
        """
        try:
            messages = self.create_prompt(query, knowledge_base_context, search_results_context)
            
            logger.info(f"Generating response for query: '{query[:100]}...'")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Log token usage
            usage = response.usage
            logger.info(f"Response generated. Tokens used - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    async def generate_response_stream(
        self,
        query: str,
        knowledge_base_context: List[Dict],
        search_results_context: Optional[str] = None
    ):
        """
        Generate a streaming response using the LLM.
        
        Args:
            query: User's question
            knowledge_base_context: Results from vector store
            search_results_context: Formatted web search results
            
        Yields:
            Chunks of the generated response
        """
        try:
            messages = self.create_prompt(query, knowledge_base_context, search_results_context)
            
            logger.info(f"Generating streaming response for query: '{query[:100]}...'")
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming LLM response: {str(e)}")
            raise

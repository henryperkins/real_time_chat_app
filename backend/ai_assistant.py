import asyncio
import logging
from typing import Dict, List, Optional, Any
from azure_openai_config import azure_openai

# Set up logging
logger = logging.getLogger(__name__)


class AIAssistant:
    """Handles AI assistant interactions using Azure OpenAI."""

    def __init__(self):
        """Initialize AI assistant."""
        self.client = azure_openai

    async def get_ai_response(
        self,
        message: str,
        conversation_history: List[str],
        project_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a response from the AI assistant.
        
        Args:
            message: The user's message
            conversation_history: List of previous messages
            project_id: Optional project ID for context
            reasoning_effort: Optional reasoning effort level 
                (low/medium/high)
            
        Returns:
            Dict containing:
                - content: The AI's response text
                - usage: Token usage statistics
                - content_filter_results: Content filter results
                - finish_reason: Reason for completion
                - model: Model used for response
        """
        try:
            # Get deployment based on purpose
            purpose = 'chat' if project_id else 'default'

            # Construct messages list
            messages = []
            
            # Add system message (will be converted to developer message for o1 models)
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
            
            # Add conversation history
            for msg in conversation_history:
                messages.append({
                    "role": "user",
                    "content": msg
                })
                
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })

            # Call Azure OpenAI API with enhanced parameters
            response = await self.client.generate_chat_completion(
                messages=messages,
                purpose=purpose,
                reasoning_effort=reasoning_effort,
                # Reasonable default for chat
                max_completion_tokens=2000
            )
            
            return {
                "content": response["content"],
                "usage": response.get("usage", {}),
                "content_filter_results": response.get("content_filter_results"),
                "finish_reason": response.get("finish_reason"),
                "model": response.get("model")
            }

        except Exception as e:
            error_msg = f"Error in AI response: {str(e)}"
            logger.error(error_msg)
            return {
                "content": (
                    "Error: Unable to process the request. "
                    f"Details: {str(e)}"
                ),
                "error": True
            }

    def get_ai_response_sync(
        self,
        message: str,
        conversation_history: List[str],
        project_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for get_ai_response.
        
        Args:
            message: The user's message
            conversation_history: List of previous messages
            project_id: Optional project ID for context
            reasoning_effort: Optional reasoning effort level
            
        Returns:
            Dict containing response data and metadata
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self.get_ai_response(
                    message=message,
                    conversation_history=conversation_history,
                    project_id=project_id,
                    reasoning_effort=reasoning_effort
                )
            )
            loop.close()
            return response
        except Exception as e:
            error_msg = f"Error in sync AI response: {str(e)}"
            logger.error(error_msg)
            return {
                "content": (
                    "Error: Unable to process the request. "
                    f"Details: {str(e)}"
                ),
                "error": True
            }

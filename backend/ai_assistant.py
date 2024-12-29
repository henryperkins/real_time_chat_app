import asyncio
import logging
from typing import Dict, List, Optional, Any
from azure_openai_config import azure_openai
<<<<<<< HEAD

# Set up logging
logger = logging.getLogger(__name__)

=======
import logging
from models import Project
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

class AIAssistant:
    """Handles AI assistant interactions using Azure OpenAI."""

    def __init__(self):
<<<<<<< HEAD
        """Initialize AI assistant."""
        self.client = azure_openai
=======
        """Initialize the AI assistant with Azure OpenAI client."""
        if azure_openai is None or azure_openai.client is None:
            raise RuntimeError(
                "AzureOpenAIConfig is not initialized. Please check the configuration, "
                "network connectivity, and environment variables. Ensure that the "
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY are correctly set in the .env file."
            )
        self.client = azure_openai.client
        self.logger = logging.getLogger(__name__)
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

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
        if azure_openai is None or azure_openai.client is None:
            self.logger.error("AzureOpenAIConfig is not initialized. Cannot generate AI response.")
            return "Error: AI assistant is currently unavailable. Please try again later."

        try:
            self.logger.info(f"Generating AI response for message: {message}")
            self.logger.info(f"Conversation history: {conversation_history}")
            # Get deployment based on purpose and project's language model
            purpose = 'chat' if project_id else 'default'
<<<<<<< HEAD
=======
            if project_id:
                project = Project.query.get(project_id)
                model = project.language_model
            else:
                model = 'gpt-4'  # Default model if no project specified
            deployment = azure_openai.get_deployment(purpose)
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

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
<<<<<<< HEAD
            
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
=======

            # Call the Azure OpenAI API with appropriate parameters
            if deployment.model == "o1-preview":
                response = self.client.chat.completions.create(
                    model=deployment.name,
                    messages=messages,
                    max_tokens=deployment.max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=deployment.name,
                    messages=messages,
                    max_tokens=deployment.max_tokens
                )

            ai_response = response.choices[0].message.content.strip()
            self.logger.info(f"Raw AI response: {response.choices[0].message.content}")
            self.logger.info(f"AI response generated: {ai_response}")
            return ai_response

        except Exception as e:
            self.logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
            return f"Error: Unable to process the request. Details: {str(e)}"
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

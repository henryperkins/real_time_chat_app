from azure_openai_config import azure_openai
import logging
from models import Project

class AIAssistant:
    """Handles AI assistant interactions using Azure OpenAI."""

    def __init__(self):
        """Initialize the AI assistant with Azure OpenAI client."""
        if azure_openai is None or azure_openai.client is None:
            raise RuntimeError(
                "AzureOpenAIConfig is not initialized. Please check the configuration, "
                "network connectivity, and environment variables. Ensure that the "
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY are correctly set in the .env file."
            )
        self.client = azure_openai.client
        self.logger = logging.getLogger(__name__)

    def get_ai_response(self, message, conversation_history, project_id=None):
        """
        Get a response from the AI assistant.
        
        Args:
            message: The user's message
            conversation_history: List of previous messages
            project_id: Optional project ID for context
            
        Returns:
            str: The AI's response
        """
        if azure_openai is None or azure_openai.client is None:
            self.logger.error("AzureOpenAIConfig is not initialized. Cannot generate AI response.")
            return "Error: AI assistant is currently unavailable. Please try again later."

        try:
            self.logger.info(f"Generating AI response for message: {message}")
            self.logger.info(f"Conversation history: {conversation_history}")
            # Get deployment based on purpose and project's language model
            purpose = 'chat' if project_id else 'default'
            if project_id:
                project = Project.query.get(project_id)
                model = project.language_model
            else:
                model = 'gpt-4'  # Default model if no project specified
            deployment = azure_openai.get_deployment(purpose)

            # Construct messages list
            messages = []
            if deployment.model != "o1-preview":
                # Add system message for models that support it
                messages.append(
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }
                )
            
            # Add conversation history
            for msg in conversation_history:
                messages.append(
                    {
                        "role": "user",
                        "content": msg
                    }
                )
                
            # Add current message
            messages.append(
                {
                    "role": "user",
                    "content": message
                }
            )

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

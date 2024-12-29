from azure_openai_config import azure_openai
import logging

class AIAssistant:
    """Handles AI assistant interactions using Azure OpenAI."""

    def __init__(self):
        """Initialize the AI assistant with Azure OpenAI client."""
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
        try:
            self.logger.info(f"Generating AI response for message: {message}")
            self.logger.info(f"Conversation history: {conversation_history}")
            # Get deployment based on purpose
            purpose = 'chat' if project_id else 'default'
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

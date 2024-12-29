from azure_openai_config import azure_openai


class AIAssistant:
    """Handles AI assistant interactions using Azure OpenAI."""

    def __init__(self):
        """Initialize the AI assistant with Azure OpenAI client."""
        self.client = azure_openai.client

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
                    max_completion_tokens=deployment.max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=deployment.name,
                    messages=messages,
                    max_tokens=deployment.max_tokens
                )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: Unable to process the request. Details: {str(e)}"

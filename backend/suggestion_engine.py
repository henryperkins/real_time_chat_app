from ai_assistant import AIAssistant


class SuggestionEngine:
    def __init__(self, ai_assistant: AIAssistant):
        self.ai_assistant = ai_assistant

    def get_suggestions(self, partial_message, conversation_history, project_id=None, max_suggestions=3):
        """
        Generates suggestions based on partial message, using RAG context if a project_id is passed,
        otherwise based on the conversation history.
        """
        try:
            # Construct a prompt for suggestions
            if project_id:
                prompt = f"Project Context: {conversation_history}\nUser is typing: {partial_message}\nAI Suggestions:"
            else:
                prompt = f"Conversation History: {conversation_history}\nUser is typing: {partial_message}\nAI Suggestions:"

            # Call the AI assistant to generate suggestions
            response = self.ai_assistant.get_ai_response(
                message=prompt,
                conversation_history=conversation_history,
                project_id=project_id)

            # Split the response into individual suggestions
            suggestions = response.split("\n")[:max_suggestions]
            return [s.strip() for s in suggestions if s.strip()]

        except Exception as e:
            # Handle errors gracefully
            return [f"Error generating suggestions: {str(e)}"]

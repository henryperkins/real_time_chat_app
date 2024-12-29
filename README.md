<<<<<<< HEAD
# Real-Time Chat App

A real-time chat application with AI assistance powered by Azure OpenAI.

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Configure environment variables:
- Copy `.env.template` to `.env`
- Update with your Azure OpenAI configuration
- Optional: Configure multiple deployments in `AZURE_OPENAI_DEPLOYMENTS`

3. Run the application:
```bash
cd backend
python app.py
```

## Azure OpenAI Integration

The application uses Azure OpenAI for AI-assisted chat features:

- Supports multiple model deployments for different purposes
- Handles both standard GPT models and o1-series reasoning models
- Provides token usage tracking and content filtering
- Supports advanced features like reasoning effort control

### Configuration

Configure Azure OpenAI in your `.env` file:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

For multiple deployments, configure the `AZURE_OPENAI_DEPLOYMENTS` variable:

```json
[
  {
    "name": "gpt4-deployment",
    "model": "gpt-4",
    "purpose": "default",
    "max_tokens": 4000
  },
  {
    "name": "o1-deployment", 
    "model": "o1",
    "purpose": "chat",
    "max_tokens": 2000
  }
]
```

### Features

1. **Multiple Model Support**
   - Configure different models for different purposes
   - Automatic handling of model-specific requirements
   - Support for both standard and o1-series models

2. **O1 Series Integration**
   - Automatic conversion of system messages to developer messages
   - Support for reasoning effort control (low/medium/high)
   - Proper handling of max_completion_tokens parameter

3. **Enhanced Response Data**
   - Token usage tracking including cached and reasoning tokens
   - Content filtering results
   - Model information and completion status

4. **Error Handling**
   - Comprehensive error logging
   - Graceful error recovery
   - Detailed error messages for debugging

### Usage Example

```python
from ai_assistant import AIAssistant

# Initialize assistant
assistant = AIAssistant()

# Get AI response with reasoning control
response = await assistant.get_ai_response(
    message="What are the key principles of software design?",
    conversation_history=[],
    reasoning_effort="high"  # Optional: Control reasoning depth
)

# Access response data
print(f"AI Response: {response['content']}")
print(f"Token Usage: {response['usage']}")
print(f"Model Used: {response['model']}")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# Real-Time Chat Application Setup

## Environment Variables Setup

To run this application, you need to set up the following environment variables:

1. **Using a `.env` file**:
   - Create a `.env` file in the root directory of your project.
   - Add the following lines to the `.env` file:
     ```
     AZURE_OPENAI_ENDPOINT=your_endpoint_here
     AZURE_OPENAI_API_KEY=your_api_key_here
     ```
   - Replace `your_endpoint_here` and `your_api_key_here` with your actual Azure OpenAI endpoint and API key.

2. **Setting environment variables directly in the terminal**:
   - Before running your application, set the environment variables in your terminal:
     - On Unix/Linux/Mac:
       ```bash
       export AZURE_OPENAI_ENDPOINT=your_endpoint_here
       export AZURE_OPENAI_API_KEY=your_api_key_here
       ```
     - On Windows (Command Prompt):
       ```cmd
       set AZURE_OPENAI_ENDPOINT=your_endpoint_here
       set AZURE_OPENAI_API_KEY=your_api_key_here
       ```
     - On Windows (PowerShell):
       ```powershell
       $env:AZURE_OPENAI_ENDPOINT='your_endpoint_here'
       $env:AZURE_OPENAI_API_KEY='your_api_key_here'
       ```

## Running the Application

After setting the environment variables, run your application:

>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

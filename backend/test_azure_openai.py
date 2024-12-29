"""Test script for Azure OpenAI integration."""
import os
import json
import asyncio
from azure_openai_config import azure_openai
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Configure deployments
DEPLOYMENTS = [
    {
        "name": "gpt-4op-deployment",
        "model": "gpt-4",
        "purpose": "default",
        "max_tokens": 2000
    },
    {
        "name": "o1-preview",
        "model": "o1-preview",
        "purpose": "chat",
        "max_tokens": 4000
    }
]

# Set deployments in environment
os.environ['AZURE_OPENAI_DEPLOYMENTS'] = json.dumps(DEPLOYMENTS)


async def test_azure_openai():
    """Test Azure OpenAI chat completion."""
    try:
        # List available deployments
        deployments = azure_openai.list_deployments()
        print("\nAvailable deployments:")
        for deployment in deployments:
            print(f"- {deployment['name']} ({deployment['model']}) "
                  f"for {deployment['purpose']}")

        # Test GPT-4 deployment with system message
        print("\nTesting GPT-4 deployment...")
        gpt4_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ]
        
        response = await azure_openai.generate_chat_completion(
            messages=gpt4_messages,
            purpose='default',
            max_tokens=100,
            temperature=0.7
        )
        
        print("\nGPT-4 Response:", response["content"])
        print("Deployment:", response["deployment"])
        print("Model:", response["model"])
        print("Purpose:", response["purpose"])
        print("Usage:", response["usage"])

        # Test o1-preview deployment with a more specific prompt
        print("\nTesting o1-preview deployment...")
        o1_messages = [
            {"role": "user", "content": "What steps should I think about when writing my first Python API?"}
        ]
        
        chat_response = await azure_openai.generate_chat_completion(
            messages=o1_messages,
            purpose='chat',
            max_completion_tokens=5000  # Using larger token limit as shown in example
        )
        
        print("\no1-preview Response:", chat_response["content"])
        print("Deployment:", chat_response["deployment"])
        print("Model:", chat_response["model"])
        print("Purpose:", chat_response["purpose"])
        print("Usage:", chat_response["usage"])
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_azure_openai())

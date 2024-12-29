"""Test script for Azure OpenAI integration."""
import os
import json
import asyncio
import logging
from azure_openai_config import azure_openai, AzureOpenAIError
from dotenv import load_dotenv
<<<<<<< HEAD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
=======
from models import db, Project
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

# Load environment variables
load_dotenv()

# Configure deployments
DEPLOYMENTS = [
    {
        "name": os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4'),
        "model": "gpt-4",
        "purpose": "default",
        "max_tokens": 2000
    }
]

# Optional: Add o1 deployment if configured
O1_DEPLOYMENT = os.getenv('AZURE_OPENAI_O1_DEPLOYMENT')
if O1_DEPLOYMENT:
    DEPLOYMENTS.append({
        "name": O1_DEPLOYMENT,
        "model": "o1",
        "purpose": "chat",
        "max_tokens": 4000
    })


# Set deployments in environment
os.environ['AZURE_OPENAI_DEPLOYMENTS'] = json.dumps(DEPLOYMENTS)


async def test_deployment(
    purpose: str,
    messages: list,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None
) -> bool:
    """Test a specific deployment configuration."""
    try:
        logger.info(f"\nTesting {purpose} deployment...")
        
        response = await azure_openai.generate_chat_completion(
            messages=messages,
            purpose=purpose,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )
        
        # Print response details
        logger.info(f"\nResponse from {purpose}:")
        logger.info(f"Content: {response['content']}")
        logger.info(f"Deployment: {response['deployment']}")
        logger.info(f"Model: {response['model']}")
        logger.info(f"Purpose: {response['purpose']}")
        logger.info(f"Usage: {response['usage']}")
        
        # Log content filter results if available
        if response.get('content_filter_results'):
            logger.info(
                "Content Filter Results: %s",
                response['content_filter_results']
            )
        
        # Log reasoning tokens if available
        if response['usage'].get('reasoning_tokens'):
            logger.info(
                "Reasoning Tokens: %d",
                response['usage']['reasoning_tokens']
            )
        
        return True
        
    except AzureOpenAIError as e:
        logger.error(f"Azure OpenAI Error: {str(e)}")
        if hasattr(e, 'error_type'):
            logger.error(f"Error Type: {e.error_type}")
        if hasattr(e, 'retry_after'):
            logger.error(f"Retry After: {e.retry_after} seconds")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False



async def test_azure_openai():
    """Test Azure OpenAI chat completion."""
    try:
        # Check environment configuration
        logger.info("\nEnvironment Configuration:")
        logger.info(
            f"AZURE_OPENAI_ENDPOINT: "
            f"{'Set' if os.getenv('AZURE_OPENAI_ENDPOINT') else 'Not Set'}"
        )
        logger.info(
            f"AZURE_OPENAI_KEY: "
            f"{'Set' if os.getenv('AZURE_OPENAI_KEY') else 'Not Set'}"
        )
        logger.info(
            f"API Version: "
            f"{os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')}"
        )

        # List available deployments
        deployments = azure_openai.list_deployments()
        logger.info("\nAvailable deployments:")
        for deployment in deployments:
            logger.info(
                f"- {deployment['name']} ({deployment['model']}) "
                f"for {deployment['purpose']}"
            )

        # Test standard GPT deployment
        gpt_success = await test_deployment(
            purpose='default',
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello! Can you help me test the API?"
                }
            ],
            max_tokens=100,
            temperature=0.7
        )

        # Test o1 deployment if available
        o1_success = True
        if O1_DEPLOYMENT:
            o1_success = await test_deployment(
                purpose='chat',
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "What are the key considerations for "
                            "designing a RESTful API?"
                        )
                    }
                ],
                max_completion_tokens=2000,
                reasoning_effort="high"
            )

        return gpt_success and o1_success

    except Exception as e:
        logger.error(f"Error in test_azure_openai: {str(e)}")
        return False


<<<<<<< HEAD
def main():
    """Run the test suite."""
    logger.info("\nStarting Azure OpenAI Integration Tests...")
    
    success = asyncio.run(test_azure_openai())
    
    if success:
        logger.info("\nAll tests completed successfully! ✅")
        exit(0)
    else:
        logger.error("\nSome tests failed! ❌")
        exit(1)

if __name__ == "__main__":
    main()
=======
async def test_azure_openai_with_project_model():
    """Test Azure OpenAI with a project-specific language model."""
    try:
        # Assume a project exists with ID 1
        project_id = 1
        project = Project.query.get(project_id)
        project.language_model = 'gpt-4o'
        db.session.commit()

        # Test with project-specific model
        print("\nTesting with project-specific model...")
        project_messages = [
            {"role": "user", "content": "What steps should I think about when writing my first Python API?"}
        ]
        
        chat_response = await azure_openai.generate_chat_completion(
            messages=project_messages,
            purpose='chat',
            max_completion_tokens=5000
        )
        
        print("\nProject-specific Response:", chat_response["content"])
        print("Deployment:", chat_response["deployment"])
        print("Model:", chat_response["model"])
        print("Purpose:", chat_response["purpose"])
        print("Usage:", chat_response["usage"])
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_azure_openai())
    asyncio.run(test_azure_openai_with_project_model())
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180

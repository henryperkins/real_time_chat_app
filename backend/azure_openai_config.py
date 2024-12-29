"""Azure OpenAI configuration and client initialization."""
from typing import Optional, Dict, Any, List
import os
import json
import logging
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()


class DeploymentConfig:
    """Configuration for an Azure OpenAI deployment."""

    def __init__(
        self,
        name: str,
        model: str,
        purpose: str,
        max_tokens: Optional[int] = None
    ):
        """Initialize deployment configuration."""
        self.name = name
        self.model = model
        self.purpose = purpose
        self.max_tokens = max_tokens


class AzureOpenAIConfig:
    """Azure OpenAI configuration and client management."""

    def __init__(self):
        """Initialize Azure OpenAI configuration."""
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = os.getenv('AZURE_OPENAI_KEY')
        self.api_version = os.getenv(
            'AZURE_OPENAI_API_VERSION',
            '2024-10-01-preview'
        )

        if not all([self.azure_endpoint, self.api_key]):
            raise ValueError(
                "Missing required Azure OpenAI configuration. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY "
                "environment variables."
            )

        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

        # Load deployment configurations
        self.deployments = self._load_deployments()
        logger.info(f"Loaded {len(self.deployments)} deployments")

    def _load_deployments(self) -> Dict[str, DeploymentConfig]:
        """Load deployment configurations from environment."""
        deployments = {}
        deployment_str = os.getenv('AZURE_OPENAI_DEPLOYMENTS')
        
        if deployment_str:
            try:
                deployment_list = json.loads(deployment_str)
                for deploy in deployment_list:
                    config = DeploymentConfig(
                        name=deploy['name'],
                        model=deploy['model'],
                        purpose=deploy['purpose'],
                        max_tokens=deploy.get('max_tokens')
                    )
                    deployments[deploy['purpose']] = config
            except Exception as e:
                logger.error(f"Error loading deployments: {str(e)}")
        
        # Add default deployment if none specified
        if not deployments:
            default_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
            deployments['default'] = DeploymentConfig(
                name=default_deployment,
                model='gpt-4',
                purpose='default'
            )
        
        return deployments

    def get_deployment(self, purpose: str = 'default') -> DeploymentConfig:
        """Get deployment configuration for a specific purpose."""
        if purpose not in self.deployments:
            logger.warning(
                f"Deployment for purpose '{purpose}' not found, "
                "using default deployment"
            )
            purpose = 'default'
        return self.deployments[purpose]

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        purpose: str = 'default',
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate a chat completion using Azure OpenAI."""
        try:
            deployment = self.get_deployment(purpose)
            
            # Handle special requirements for o1-preview model
            if deployment.model == 'o1-preview':
                # Create a new client with specific API version for o1-preview
                o1_client = AsyncAzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.api_key,
                    api_version="2024-12-01-preview"  # Updated API version for o1-preview
                )
                
                # Remove system messages as they're not supported
                messages = [msg for msg in messages if msg['role'] != 'system']
                # Format messages for o1-preview (keeping only user messages)
                messages = [{
                    'role': msg['role'],
                    'content': msg['content'].strip()
                } for msg in messages if msg['role'] != 'system']
                
                # Use the parameters as shown in the example
                params = {
                    'model': deployment.name,
                    'messages': messages,
                    'max_completion_tokens': max_completion_tokens or max_tokens or deployment.max_tokens,
                    'temperature': 1.0  # o1-preview requires temperature=1
                }
                response = await o1_client.chat.completions.create(**params)
            else:
                # Standard parameters for other models
                params = {
                    'model': deployment.name,
                    'messages': messages,
                    'max_tokens': max_tokens or deployment.max_tokens,
                    'temperature': temperature
                }
                response = await self.client.chat.completions.create(**params)
            
            # Get content from response
            content = response.choices[0].message.content or ''
            
            return {
                "content": content,
                "usage": response.usage.dict() if response.usage else None,
                "deployment": deployment.name,
                "model": deployment.model,
                "purpose": deployment.purpose
            }
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise AzureOpenAIError(str(e))

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all available deployments."""
        return [
            {
                "name": d.name,
                "model": d.model,
                "purpose": d.purpose,
                "max_tokens": d.max_tokens
            }
            for d in self.deployments.values()
        ]


class AzureOpenAIError(Exception):
    """Custom exception for Azure OpenAI-related errors."""

    def __init__(
        self,
        message: str,
        error_type: Optional[str] = None,
        retry_after: Optional[int] = None
    ) -> None:
        """Initialize the error with optional type and retry information."""
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


# Create a global instance
azure_openai = AzureOpenAIConfig()

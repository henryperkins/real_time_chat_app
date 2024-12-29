"""Azure OpenAI configuration and client initialization."""
from enum import Enum
from typing import Optional, Dict, Any, List
import os
import json
import logging
import asyncio
from openai import AsyncAzureOpenAI, OpenAIError
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
        # Get required configuration
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_KEY')
        api_version = os.getenv(
            'AZURE_OPENAI_API_VERSION',
            '2024-12-01-preview'
            '2024-12-01-preview'
        )

        if not all([azure_endpoint, api_key]):
            raise ValueError(
                "Missing required Azure OpenAI configuration. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
                "environment variables."
            )

        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,  # type: ignore
            api_key=api_key,  # type: ignore
            api_version=api_version
        )

        self.deployments = {}

    async def async_init(self):
        """Asynchronously initialize Azure OpenAI configuration."""
        self.deployments = await self._load_deployments()
        if len(self.deployments) <= 1:
            logger.warning(f"Only {len(self.deployments)} deployment(s) loaded. Expected more.")
        else:
            logger.info(f"Loaded {len(self.deployments)} deployments")

    async def fetch_models(self) -> List[Dict[str, Any]]:
        """Fetch all models available for the Azure OpenAI resource."""
        retries = 3
        for attempt in range(retries):
            try:
                response = await self.client.models.list()
                return response.data
            except Exception as e:
                logger.error(f"Error fetching models (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    logger.warning(f"Retrying connection in {2 ** attempt} seconds...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("All retries failed. Unable to fetch models.")
                    raise

    async def _load_deployments(self) -> Dict[str, DeploymentConfig]:
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
                        max_tokens=deploy.get('max_tokens', None)
                    )
                    deployments[deploy['purpose']] = config
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON for deployments: {str(e)}")
            except KeyError as e:
                logger.error(f"Missing key in deployment configuration: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error loading deployments: {str(e)}")
        
        # Fetch models to dynamically set the default
        models = await self.fetch_models()
        default_model = None
        
        if not deployments:
            # If no deployments are specified, choose the first generally available model
            for model in models:
                if model['lifecycle_status'] == 'generally-available':
                    default_model = model['id']
                    break
            
            if default_model:
                logger.warning("No deployments specified. Falling back to default configuration.")
                deployments['default'] = DeploymentConfig(
                    name=default_model,
                    model=default_model,
                    purpose='default'
                )
            else:
                logger.warning("No generally available models found. Using 'gpt-4' as default.")
                deployments['default'] = DeploymentConfig(
                    name='gpt-4',
                    model='gpt-4',
                    purpose='default'
                )
        else:
            # If deployments are specified, check if they are valid
            for purpose, config in deployments.items():
                found = False
                for model in models:
                    if model['id'] == config.model:
                        found = True
                        break
                if not found:
                    logger.warning(f"Deployment model '{config.model}' not found. Using 'gpt-4' as default for purpose '{purpose}'.")
                    deployments[purpose] = DeploymentConfig(
                        name='gpt-4',
                        model='gpt-4',
                        purpose=purpose
                    )
        
        return deployments

    def get_deployment(self, purpose: str = 'default') -> DeploymentConfig:
        """Get deployment configuration for a specific purpose."""
        if purpose not in self.deployments:
            logger.warning((
                f"Deployment for purpose '{purpose}' not found, "
                "using default deployment"
            ))
            purpose = 'default'
        return self.deployments[purpose]

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        purpose: str = 'default',
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None
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
            
            # Extract content filter results if available
            content_filter_results = (
                response.choices[0].content_filter_results
                if hasattr(response.choices[0], 'content_filter_results')
                else None
            )
            
            # Extract usage details
            usage_data = {}
            if response.usage:
                usage_dict = response.usage.dict()
                usage_data = {
                    'prompt_tokens': usage_dict.get('prompt_tokens', 0),
                    'completion_tokens': usage_dict.get('completion_tokens', 0),
                    'total_tokens': usage_dict.get('total_tokens', 0),
                }
                
                # Get completion tokens details
                completion_details = usage_dict.get(
                    'completion_tokens_details',
                    {}
                )
                if completion_details:
                    usage_data['reasoning_tokens'] = (
                        completion_details.get('reasoning_tokens', 0)
                    )
                
                # Get prompt tokens details
                prompt_details = usage_dict.get('prompt_tokens_details', {})
                if prompt_details:
                    usage_data['cached_tokens'] = (
                        prompt_details.get('cached_tokens', 0)
                    )

            # Return formatted response
            result = {
                "content": response.choices[0].message.content or '',
                "usage": usage_data,
                "deployment": deployment.name,
                "model": deployment.model,
                "purpose": deployment.purpose,
                "content_filter_results": content_filter_results,
                "finish_reason": response.choices[0].finish_reason
            }
            return result

        except OpenAIError as e:
            error_msg = f"Error in chat completion: {str(e)}"
            logger.error(error_msg)
            
            # Map error type based on the error message and type
            error_type = None
            retry_after = None
            content_filter_results = None
            
            error_message = str(e).lower()
            if "rate limit" in error_message:
                error_type = OpenAIErrorType.RATE_LIMIT
            elif "quota exceeded" in error_message:
                error_type = OpenAIErrorType.QUOTA_EXCEEDED
            elif "invalid request" in error_message:
                error_type = OpenAIErrorType.INVALID_REQUEST
            elif "content filter" in error_message:
                error_type = OpenAIErrorType.RESPONSIBLE_AI_POLICY_VIOLATION
            
            # Try to extract additional error details
            try:
                if hasattr(e, 'body'):
                    body = e.body
                    if isinstance(body, dict):
                        error_data = body.get('error', {})
                        if error_data.get('type'):
                            try:
                                error_type = OpenAIErrorType(error_data['type'])
                            except ValueError:
                                pass
                        
                        inner_error = error_data.get('inner_error', {})
                        if inner_error:
                            content_filter_results = inner_error.get(
                                'content_filter_results'
                            )
            except Exception as parse_error:
                logger.warning(
                    f"Failed to parse error details: {parse_error}"
                )
            
            raise AzureOpenAIError(
                error_msg,
                error_type=error_type,
                retry_after=retry_after,
                content_filter_results=content_filter_results
            ) from e
        except Exception as e:
            # Handle non-OpenAI errors
            error_msg = f"Unexpected error in chat completion: {str(e)}"
            logger.error(error_msg)
            raise AzureOpenAIError(error_msg) from e

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


class OpenAIErrorType(Enum):
    RATE_LIMIT = "rate_limit_error"
    QUOTA_EXCEEDED = "quota_exceeded_error"
    INVALID_REQUEST = "invalid_request_error"
    API_ERROR = "api_error"
    RESPONSIBLE_AI_POLICY_VIOLATION = "ResponsibleAIPolicyViolation"


class AzureOpenAIError(Exception):
    """Custom exception for Azure OpenAI-related errors."""

    def __init__(
        self,
        message: str,
        error_type: Optional[OpenAIErrorType] = None,
        retry_after: Optional[int] = None,
        content_filter_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the error with optional type and retry information."""
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after
        self.content_filter_results = content_filter_results

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            "message": str(self),
            "error_type": self.error_type.value if self.error_type else None,
            "retry_after": self.retry_after,
            "content_filter_results": self.content_filter_results
        }


# Create a global instance
azure_openai = None  # This will be initialized asynchronously in app.py

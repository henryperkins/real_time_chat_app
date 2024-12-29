"""Azure OpenAI configuration and client initialization."""
from enum import Enum
from typing import Optional, Dict, Any, List
import os
import json
import logging
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
        )

        if not all([azure_endpoint, api_key]):
            raise ValueError(
                "Missing required Azure OpenAI configuration. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY "
                "environment variables."
            )

        # Store configuration after validation
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version

        # Initialize client with validated configuration
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,  # type: ignore
            api_key=api_key,  # type: ignore
            api_version=api_version
        )

        # Load deployment configurations
        self.deployments = self._load_deployments()
        logger.info(
            f"Loaded {len(self.deployments)} deployments"
        )

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
                        max_tokens=deploy.get('max_tokens', None)
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
            
            # Determine if this is an o1 series model
            is_o1_model = any(
                deployment.model.startswith(prefix)
                for prefix in ['o1-', 'o1']
            )
            
            # Format messages appropriately
            formatted_messages = []
            for msg in messages:
                if is_o1_model and msg['role'] == 'system':
                    # Convert system messages to developer messages for o1 models
                    formatted_messages.append({
                        'role': 'developer',
                        'content': msg['content'].strip()
                    })
                else:
                    formatted_messages.append({
                        'role': msg['role'],
                        'content': msg['content'].strip()
                    })
            
            # Build parameters based on model type
            params = {
                'model': deployment.name,
                'messages': formatted_messages,
            }
            
            if is_o1_model:
                # o1 series specific parameters
                params['max_completion_tokens'] = (
                    max_completion_tokens or
                    max_tokens or
                    deployment.max_tokens
                )
                params['temperature'] = 1.0  # Required for o1 series
                if reasoning_effort:
                    params['reasoning_effort'] = reasoning_effort
            else:
                # Standard parameters for other models
                params['max_tokens'] = max_tokens or deployment.max_tokens
                if temperature is not None:
                    params['temperature'] = temperature
            
            # Make API call
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
azure_openai = AzureOpenAIConfig()

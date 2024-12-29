# Guide for Azure OpenAI
## 1. Authentication Guide for Azure OpenAI API

**Overview:**

To interact with the Azure OpenAI API, you need to authenticate your requests using an API key. This guide will walk you through setting up your environment and authenticating your API requests. This version of the API uses the `2024-10-01-preview` by default.

**Initialize the Client:**

Use the `AzureOpenAI` client for synchronous operations or `AsyncAzureOpenAI` for asynchronous operations. Ensure the client is initialized with the correct endpoint and API key, along with the desired `api_version`.

```python
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-01-preview')

# Initialize the client
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)
```

**Validate the Connection:**

Make a simple API call to validate your connection and ensure everything is set up correctly.

```python
response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=10
)
print("Connection successful!")
print(response.choices[0].message.content)
```

### Debugging Tips

* **Check Environment Variables:** Ensure all necessary environment variables are correctly set and accessible in your script.
* **API Key Validity:** Verify that your API key is active and has the necessary permissions.
* **Endpoint URL:** Double-check the endpoint URL to ensure it matches your Azure OpenAI resource.
* **API Version:** Ensure the `api_version` is set to the correct version of the API you intend to use. The default value is `2024-10-01-preview`.
* **Error Handling:** Implement error handling to capture and log any issues during the authentication process.

### References

* [Azure OpenAI Quickstart Guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart)
* [Python Dotenv Documentation](https://pypi.org/project/python-dotenv/)
* [Switching between Azure OpenAI and OpenAI endpoints](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints?source=recommendations)

## 2. Function Calling with Error Handling (Updated)

**Overview:**

Enhanced function calling implementation with improved tool handling and response validation. This version of the API uses the `tools` parameter instead of the deprecated `functions` and `function_call` to describe tools available for the model.

```python
import json
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

class ToolManager:
    def __init__(self):
        self.available_tools: Dict[str, callable] = {}
        self.tool_schemas: Dict[str, Dict] = {}

    def register_tool(self, name: str, func: callable, schema: Dict):
        """Register a tool with its schema."""
        self.available_tools[name] = func
        self.tool_schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": schema.get("description", ""),
                "parameters": schema
            }
        }

    def get_tool_definitions(self) -> List[Dict]:
        """Get all tool definitions for API request."""
        return list(self.tool_schemas.values())

    async def execute_tool(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool and return its result."""
        tool_name = tool_call.get("function", {}).get("name")
        if tool_name not in self.available_tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        arguments = json.loads(tool_call["function"]["arguments"])
        tool_func = self.available_tools[tool_name]
        return await tool_func(**arguments)

class EnhancedFunctionCaller:
    def __init__(self, tool_manager: ToolManager, client: AzureOpenAI):
        self.tool_manager = tool_manager
        self.client = client

    async def process_with_tools(
        self,
        messages: List[Dict[str, str]],
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """Process a request with potential tool calls."""
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=messages,
                tools=self.tool_manager.get_tool_definitions(),
                tool_choice=tool_choice
            )

            if response.choices[0].message.tool_calls:
                tool_messages = [response.choices[0].message.dict()]

                for tool_call in response.choices[0].message.tool_calls:
                    tool_result = await self.tool_manager.execute_tool(tool_call.dict())
                    tool_messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result),
                        "tool_call_id": tool_call.id
                    })

                # Get final response with tool results
                final_response = await self.client.chat.completions.acreate(
                    model="gpt-4",
                    messages=messages + tool_messages
                )
                return final_response.choices[0].message.dict()

            return response.choices[0].message.dict()

        except Exception as e:
            print(f"Error in function calling: {str(e)}")
            raise

# Example usage
tool_manager = ToolManager()

# Register weather tool
weather_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
        }
    },
    "required": ["location"]
}


async def get_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    # Implement actual weather API call
    return {"temperature": 20, "units": units, "location": location}


tool_manager.register_tool("get_weather", get_weather, weather_schema)
function_caller = EnhancedFunctionCaller(tool_manager, client)
```

**Reference:**

* [How to use function calling with Azure OpenAI Service - Azure OpenAI Service | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling#single-toolfunction-calling-example)
* [Working with Functions](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Functions/working_with_functions.ipynb)

## 3. Structured Output Generation (Updated)

**Overview:**

Extract structured data from text using predefined JSON schemas with the `response_format` parameter. This method guarantees that the model output adheres to the defined structure. This works with all GPT-3.5 Turbo models newer than `gpt-3.5-turbo-1106`, as well as all GPT-4 models.

```python
import json
from typing import Dict, Any, Optional
from openai import AzureOpenAI

def get_structured_output(prompt: str, schema: dict, output_type: str = "json_object"):
    messages = [
        {"role": "system", "content": "Extract information according to the provided schema."},
        {"role": "user", "content": prompt}
    ]

    response_format = {
        "type": output_type
    }
    if output_type == "json_object" and schema:
        response_format["json_schema"] = schema

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        response_format=response_format,
    )
    return json.loads(response.choices[0].message.content)


# Enhanced schema example
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"}
            }
        }
    },
    "required": ["name", "age"]
}

```

**Reference:**

[Structured Output Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?utm_source=chatgpt.com&tabs=python#getting-started)

## 4. Token Management and Cost Optimization

**Overview:**

Manage token usage effectively to optimize costs and ensure efficient API usage. It is important to consider tokens in the prompt when making calls to the API (specially using `chat/completions` requests), and to limit the number of tokens generated in the response, using `max_tokens`. The Assistants API also supports `max_prompt_tokens` and `max_completion_tokens` parameters, that should be used when possible to optimize the costs.

```python
from tiktoken import encoding_for_model

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = encoding_for_model(model)
    return len(encoding.encode(text))

def optimize_prompt(text: str, max_tokens: int = 4000):
    current_tokens = estimate_tokens(text)
    if current_tokens > max_tokens:
        # Implement truncation strategy
        encoding = encoding_for_model("gpt-4")
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text

# Usage example with token management
def managed_completion(prompt: str, max_tokens: int = 4000):
    optimized_prompt = optimize_prompt(prompt, max_tokens)
    estimated_cost = estimate_tokens(optimized_prompt) * 0.00002  # Example rate

    print(f"Estimated tokens: {estimate_tokens(optimized_prompt)}")
    print(f"Estimated cost: ${estimated_cost:.4f}")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": optimized_prompt}],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content
```

**Reference:**

[Token Management Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/manage-tokens)

## 5. Error Handling and Monitoring (Updated)

**Overview:**

Implement comprehensive error handling with specific error types and enhanced monitoring. This new API introduces richer error details, particularly for content filtering and rate limiting.

```python
from enum import Enum
from typing import Optional, Dict, Any
from openai import AzureOpenAI
from pydantic import BaseModel

class OpenAIErrorType(Enum):
    RATE_LIMIT = "rate_limit_error"
    QUOTA_EXCEEDED = "quota_exceeded_error"
    INVALID_REQUEST = "invalid_request_error"
    API_ERROR = "api_error"
    RESPONSIBLE_AI_POLICY_VIOLATION = "ResponsibleAIPolicyViolation"


class OpenAIErrorHandler:
    def __init__(self):
        self.error_counts: Dict[str, int] = {error.value: 0 for error in OpenAIErrorType}

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        error_response = {
            "success": False,
            "error_type": None,
            "message": str(error),
            "retry_after": None,
            "content_filter_results": None
        }

        if hasattr(error, 'response'):
            error_data = error.response.json().get('error', {})
            error_type = error_data.get('type')
            inner_error = error_data.get('inner_error', {})
            
            if inner_error and inner_error.get('content_filter_results'):
                error_response["content_filter_results"] = inner_error.get('content_filter_results')

            if error_type:
                self.error_counts[error_type] += 1
                error_response["error_type"] = error_type

                if error_type == OpenAIErrorType.RATE_LIMIT.value:
                    error_response["retry_after"] = error.response.headers.get('Retry-After')
        
            if error_type == OpenAIErrorType.RESPONSIBLE_AI_POLICY_VIOLATION.value:
                if inner_error and inner_error.get('content_filter_results'):
                    error_response["content_filter_results"] = inner_error.get('content_filter_results')
        return error_response


class EnhancedOpenAIMonitor:
    def __init__(self):
        self.request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0
        }
        self.error_handler = OpenAIErrorHandler()

    def log_request(self, response: Optional[Dict] = None, error: Optional[Exception] = None):
        self.request_metrics["total_requests"] += 1

        if error:
            self.request_metrics["failed_requests"] += 1
            return self.error_handler.handle_error(error)

        if response:
            self.request_metrics["successful_requests"] += 1
            if 'usage' in response:
                self.request_metrics["total_tokens"] += response['usage'].get('total_tokens', 0)
                self.request_metrics["completion_tokens"] += response['usage'].get('completion_tokens', 0)
                self.request_metrics["prompt_tokens"] += response['usage'].get('prompt_tokens', 0)
                if 'prompt_tokens_details' in response['usage'] and response['usage']['prompt_tokens_details'] and response['usage']['prompt_tokens_details'].get("cached_tokens"):
                    self.request_metrics["cached_tokens"] += response['usage']['prompt_tokens_details'].get("cached_tokens", 0)
                if 'completion_tokens_details' in response['usage'] and response['usage']['completion_tokens_details'] and response['usage']['completion_tokens_details'].get('reasoning_tokens'):
                  self.request_metrics["reasoning_tokens"] += response['usage']['completion_tokens_details'].get("reasoning_tokens", 0)

        return {"success": True}
    
    def get_metrics_summary(self):
      return self.request_metrics


import asyncio

async def robust_completion_with_monitoring(
    prompt: str,
    monitor: EnhancedOpenAIMonitor,
    client: AzureOpenAI,
    retries: int = 3
) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            response = await client.chat.completions.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            monitor.log_request(response=response.dict())
            return {
                "content": response.choices[0].message.content,
                "usage": response.usage.dict()
            }
        except Exception as e:
            error_response = monitor.log_request(error=e)
            if attempt == retries - 1:
                return error_response
            await asyncio.sleep(2 ** attempt)

```

**Reference:**

[Error Handling Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)

## 6. Batch Processing with Rate Limiting (Updated)

**Overview:**

Enhanced batch processing with improved rate limiting and quota management based on the latest Azure OpenAI API specifications. It's important to note that the rate limits depend on the specific model and API being used. Refer to the official Azure OpenAI documentation for the most up-to-date information on your specific resource's limits. This example uses a Rate Limiter class with configuration to manage requests per minute and tokens per minute.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60 # Example, adjust for your specific API tier and model
    tokens_per_minute: int = 40000 # Example, adjust for your specific API tier and model
    max_retries: int = 3
    retry_delay: float = 1.0

class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_timestamps: List[datetime] = []
        self.token_usage: List[tuple[datetime, int]] = []
        self.semaphore = asyncio.Semaphore(config.requests_per_minute)

    async def acquire(self, estimated_tokens: int = 0) -> bool:
        """Acquire permission to make a request"""
        now = datetime.now()

        # Clean old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if ts > now - timedelta(minutes=1)
        ]
        self.token_usage = [
            (ts, tokens) for ts, tokens in self.token_usage
            if ts > now - timedelta(minutes=1)
        ]

        # Check limits
        if len(self.request_timestamps) >= self.config.requests_per_minute:
            return False

        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.config.tokens_per_minute:
            return False

        async with self.semaphore:
            self.request_timestamps.append(now)
            if estimated_tokens > 0:
                self.token_usage.append((now, estimated_tokens))
            return True


class EnhancedBatchProcessor:
    def __init__(
        self,
        rate_limiter: RateLimiter,
        client: AzureOpenAI,
        max_concurrent: int = 5
    ):
        self.rate_limiter = rate_limiter
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = client
        self.results: List[Dict[str, Any]] = []

    async def process_item(
        self,
        item: str,
        estimated_tokens: int = 1000
    ) -> Dict[str, Any]:
        async with self.semaphore:
            for attempt in range(self.rate_limiter.config.max_retries):
                if await self.rate_limiter.acquire(estimated_tokens):
                    try:
                        response = await self.client.chat.completions.acreate(
                            model="gpt-4",
                            messages=[{"role": "user", "content": item}]
                        )
                        return {
                            "input": item,
                            "output": response.choices[0].message.content,
                            "usage": response.usage.dict()
                        }
                    except Exception as e:
                        if "rate_limit" in str(e).lower():
                            await asyncio.sleep(
                                self.rate_limiter.config.retry_delay * (2 ** attempt)
                            )
                            continue
                        return {"input": item, "error": str(e)}
                await asyncio.sleep(self.rate_limiter.config.retry_delay)
            return {"input": item, "error": "Rate limit exceeded after retries"}

    async def process_batch(
        self,
        items: List[str],
        estimated_tokens_per_item: int = 1000
    ) -> List[Dict[str, Any]]:
        tasks = [
            self.process_item(item, estimated_tokens_per_item)
            for item in items
        ]
        self.results = await asyncio.gather(*tasks)
        return self.results

# Usage example
rate_limit_config = RateLimitConfig(
    requests_per_minute=60,
    tokens_per_minute=40000,
    max_retries=3,
    retry_delay=1.0
)

rate_limiter = RateLimiter(rate_limit_config)
batch_processor = EnhancedBatchProcessor(rate_limiter, client)

# Example batch processing with monitoring
async def process_with_monitoring(items: List[str]):
    start_time = datetime.now()
    results = await batch_processor.process_batch(items)

    # Process results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in successful)

    print(f"Processed {len(successful)} successfully, {len(failed)} failed")
    print(f"Total tokens used: {total_tokens}")
    print(f"Processing time: {datetime.now() - start_time}")
    return results
```

## 7. Advanced Prompt Management (Updated)

**Overview:**

Enhanced prompt management with support for different message roles, structured content, and tool interactions. This version includes `tool` as a valid message role and supports a structured content for user and system messages, with text or image parts.

```python
from enum import Enum
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class PromptTemplate:
    def __init__(
        self,
        template: str,
        required_variables: List[str],
        role: MessageRole = MessageRole.USER
    ):
        self.template = template
        self.required_variables = required_variables
        self.role = role

    def format(self, variables: Dict[str, str]) -> Dict[str, str]:
        missing = [var for var in self.required_variables if var not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return {
            "role": self.role.value,
            "content": self.template.format(**variables)
        }

class EnhancedPromptManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.system_message: Optional[str] = None

    def set_system_message(self, message: str):
        """Set a system message to be included in all prompts"""
        self.system_message = message

    def add_template(self, name: str, template: PromptTemplate):
        self.templates[name] = template

    def create_messages(
        self,
        template_name: str,
        variables: Dict[str, str],
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """Create a messages array for the API request"""
        messages = []

        if include_system and self.system_message:
            messages.append({
                "role": MessageRole.SYSTEM.value,
                "content": self.system_message
            })

        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")

        messages.append(self.templates[template_name].format(variables))
        return messages

    async def execute(
        self,
        template_name: str,
        variables: Dict[str, str],
        client: AzureOpenAI,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a prompt template with optional configuration"""
        messages = self.create_messages(template_name, variables)

        request_params = {
            "model": "gpt-4",
            "messages": messages
        }

        if config:
            request_params.update(config)

        response = await client.chat.completions.acreate(**request_params)
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.dict(),
            "finish_reason": response.choices[0].finish_reason
        }

# Example usage
prompt_manager = EnhancedPromptManager()

# Set system message
prompt_manager.set_system_message(
    "You are a helpful assistant that provides clear and concise responses."
)

# Add templates
prompt_manager.add_template(
    "summarize",
    PromptTemplate(
        "Summarize the following {document_type} in {style} style:\n\n{content}",
        ["document_type", "style", "content"]
    )
)

prompt_manager.add_template(
    "analyze",
    PromptTemplate(
        "Analyze the following {data_type} and provide {analysis_type} analysis:\n\n{content}",
        ["data_type", "analysis_type", "content"]
    )
)

# Execute with configuration
config = {
    "temperature": 0.7,
    "max_tokens": 500,
    "response_format": {"type": "json_object"}
}

# result = await prompt_manager.execute(
#    "analyze",
#     {
#        "data_type": "market data",
#        "analysis_type": "trend",
#       "content": "Your data here"
#    },
#    config
#)

```

**Reference:**

[Prompt Engineering Guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)

## 8. System Monitoring and Logging

**Overview:**

Implement system monitoring and logging to track performance, usage, and errors in your Azure OpenAI applications. This version of the API returns additional usage parameters such as `cached_tokens` and `reasoning_tokens`.

```python
import time
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class APIMetrics:
    timestamp: float
    endpoint: str
    response_time: float
    status: str
    tokens_used: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    error: Optional[str] = None

class SystemMonitor:
    def __init__(self):
        self.metrics: List[APIMetrics] = []

    def log_request(self, endpoint: str, tokens: int,
                   response_time: float, status: str,
                   prompt_tokens: Optional[int] = None,
                   completion_tokens: Optional[int] = None,
                   cached_tokens: Optional[int] = None,
                   reasoning_tokens: Optional[int] = None,
                   error: str = None):
        metric = APIMetrics(
            timestamp=time.time(),
            endpoint=endpoint,
            response_time=response_time,
            status=status,
            tokens_used=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            error=error
        )
        self.metrics.append(metric)

    def get_metrics_summary(self):
        if not self.metrics:
            return "No metrics available"

        total_requests = len(self.metrics)
        avg_response_time = sum(m.response_time for m in self.metrics) / total_requests
        total_tokens = sum(m.tokens_used for m in self.metrics)
        error_rate = len([m for m in self.metrics if m.error]) / total_requests

        return {
            "total_requests": total_requests,
            "average_response_time": avg_response_time,
            "total_tokens_used": total_tokens,
            "error_rate": error_rate
        }
    
    def get_detailed_metrics(self) -> List[APIMetrics]:
        return self.metrics


# Usage example
monitor = SystemMonitor()
# Log a request example
monitor.log_request(endpoint="chat.completions", tokens=150, response_time=0.5, status="success", prompt_tokens=100, completion_tokens=50, cached_tokens=50, reasoning_tokens=10)
print(monitor.get_metrics_summary())
```

**Reference:**

[Error Handling Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)

## 9. Dynamic Prompts with Structured Outputs and Function Calling

**Overview:**

Combine dynamic prompts with `tools` (function calling) and the `response_format` parameter to generate structured outputs using the Azure OpenAI API. This approach allows you to dynamically adjust prompts based on input conditions and ensure that the API returns data in a specified format, such as JSON. This is particularly useful for applications requiring consistent data structures, like generating documentation, extracting information, or formatting API responses. This approach works with all GPT-3.5 Turbo models newer than `gpt-3.5-turbo-1106`, as well as all GPT-4 models.

**Implementation Strategies:**

1. **Dynamic Prompt Creation:**
    * Construct prompts based on input parameters, function signatures, or user preferences.
    * Include additional context or examples to guide the model towards the desired output.

2. **Define Structured Output Schema:**
    * Specify the expected JSON structure for the output using a schema.
    * Ensure that the schema includes all necessary fields and their types.

3. **API Calls with Tool Calling:**
    * Use the `tools` parameter to define the function.
    * Use the `response_format` parameter in the API request with `type: "json_object"` or `type: "json_schema"` to enforce the structured output.

4. **Process and Validate Responses:**
    * Parse and validate the API response against the schema.
    * Implement error handling to manage exceptions during API calls, including retry mechanisms and logging.

**Example Implementation:**

```python
import os
import json
from openai import AzureOpenAI
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-01-preview')

# Initialize the client
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)


def create_dynamic_prompt(function_signature: str, additional_context: str = "") -> str:
    """Create a dynamic prompt based on the function signature and additional context."""
    prompt = f"Generate a Python docstring for the following function signature:\n\n{function_signature}\n\n"
    if additional_context:
        prompt += f"Additional context: {additional_context}\n\n"
    prompt += "The docstring should include:\n- `summary`: A concise summary.\n- `args`: A list of arguments with `name`, `type`, and `description`.\n- `returns`: A description of the return value(s).\n\n"
    return prompt


def generate_structured_docstring(function_signature: str, additional_context: str = "", model: str = "gpt-35-turbo") -> Optional[Dict[str, Any]]:
    prompt = create_dynamic_prompt(function_signature, additional_context)

    # Define the expected structure of the output
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "args": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "description"]
                }
            },
            "returns": {"type": "string"}
        },
        "required": ["summary", "args", "returns"]
    }
    
    tools=[{
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring in JSON format",
                "parameters": schema
            }
        }]
    
    try:
      response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes Python docstrings in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
             tools=tools,
             response_format={"type": "json_schema", "json_schema": {"schema": schema, "name": "generate_docstring"}},
             tool_choice= {"type": "function", "function": {"name":"generate_docstring"}}
        )

        # Process the structured response
        structured_output = json.loads(response.choices[0].message.content)
        return structured_output

    except Exception as e:
        print(f"Error during API call: {e}")
        return None


# Example usage
function_signature = "def example_function(param1: int, param2: str) -> bool:"
additional_context = "This function checks if param1 is greater than param2."
docstring = generate_structured_docstring(function_signature, additional_context)
print(json.dumps(docstring, indent=2))
```

**Use Cases:**

* **Documentation Generation:** Automatically generate structured documentation for codebases by dynamically creating prompts based on function signatures and comments.
* **Data Extraction:** Extract structured data from unstructured text by defining schemas that the output must adhere to.
* **API Response Formatting:** Ensure API responses are consistently formatted by using structured outputs in conjunction with dynamic prompts.
* **Automated Report Generation:** Create structured reports from raw data inputs by defining the desired output format.

**Key Points:**

* **Dynamic Prompt Creation:** Tailor prompts to the specific context and requirements of the task.
* **Structured Output Schema:** Define the expected JSON format using JSON Schema to ensure consistent and reliable outputs.
* **Tool Calling:** Use the `tools` and `tool_choice` parameters to instruct the model to adhere to the schema.
* **Response Format:** Use the `response_format` parameter to ensure the output is structured according to a defined JSON schema.
* **Error Handling:** Implement robust error handling, including retry mechanisms and logging, to manage exceptions and validate outputs.

**Testing and Validation:**

* **Validate JSON Outputs:** Use JSON validators to ensure the API responses match the expected schema.
* **Test with Different Models:** Experiment with different models to find the best fit for your structured output needs.
* **Logging and Monitoring:** Implement logging to capture raw API responses and monitor performance.

**References:**

* [openai/Basic_Samples/Functions/working_with_functions.ipynb at main · Azure-Samples/openai](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Functions/working_with_functions.ipynb)
* [Dynamic Prompts Example on GitHub](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Completions/completions_with_dynamic_prompt.ipynb)
* [How to use structured outputs with Azure OpenAI Service - Azure OpenAI | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python-secure#function-calling-with-structured-outputs)

## 10. Advanced RAG with Hybrid Search

**Overview:**

Combine retrieval-augmented generation (RAG) with hybrid search to enhance information retrieval and response generation. This version also includes support for integrated vectorization within the Azure Search service. This example demonstrates how to combine vector and keyword search.

```python
from azure.search.documents.models import Vector
import numpy as np
from openai import AzureOpenAI

class HybridSearchRAG:
    def __init__(self, search_client, embedding_client: AzureOpenAI):
        self.search_client = search_client
        self.embedding_client = embedding_client

    async def get_embeddings(self, text: str):
        response = await self.embedding_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    async def hybrid_search(self, query: str, top_k: int = 3):
        # Get query embedding
        query_vector = await self.get_embeddings(query)

        # Perform hybrid search
        results = self.search_client.search(
            search_text=query,
             vector_queries=[{
                "vector": query_vector,
                "k": top_k,
                 "fields": "content_vector"
            }],
            select=["content", "title"],
            top=top_k
        )

        return [{"content": doc["content"], "title": doc["title"]} for doc in results]

# Usage example
async def enhanced_rag_query(query: str, client: AzureOpenAI, search_client):
    rag = HybridSearchRAG(search_client, client)
    context_docs = await rag.hybrid_search(query)

    # Format context
    context = "\n".join([f"Title: {doc['title']}\nContent: {doc['content']}"
                         for doc in context_docs])

    # Generate response
    response = await client.chat.completions.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content
```

**Reference:**

[Hybrid Search Documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)

## 11. Advanced Content Filtering and Safety

**Overview:**

Implement advanced content filtering to ensure the safety and appropriateness of AI-generated content, using predefined categories and thresholds, as well as custom block lists. This version includes richer details on content filtering results, including specific categories, severities, and the use of citations, and information on protected and ungrounded content.

```python
from typing import List, Dict, Optional
from openai import AzureOpenAI

class ContentFilter:
    def __init__(self, client: AzureOpenAI):
        self.client = client
        self.blocked_terms = set()
        self.content_categories = {
            "hate": 0.7,
            "sexual": 0.8,
            "violence": 0.8,
            "self_harm": 0.9
        }

    def add_blocked_terms(self, terms: List[str]):
        self.blocked_terms.update(terms)

    async def check_content(self, text: str) -> Dict[str, Any]:
        # Check against blocked terms
        for term in self.blocked_terms:
            if term.lower() in text.lower():
                return {"safe": False, "reason": f"Blocked term: {term}"}

        # Use Azure Content Safety API
        try:
            response = await self.client.moderations.create(input=text)
            results = response.results[0]
            
            if results.categories:
                for category, threshold in self.content_categories.items():
                    if getattr(results.categories, category) > threshold:
                        return {
                            "safe": False,
                            "reason": f"Content filtered: {category}",
                            "categories": results.categories.dict(),
                            "category_scores": results.category_scores.dict()
                        }
            
            return {"safe": True, "reason": None}
        except Exception as e:
            print(f"Content filtering error: {e}")
            return {"safe": False, "reason": "Error in content check"}

async def safe_completion(prompt: str, content_filter: ContentFilter, client: AzureOpenAI):
    # Check input content
    input_check = await content_filter.check_content(prompt)
    if not input_check["safe"]:
        raise ValueError(f"Input content filtered: {input_check['reason']}, Categories: {input_check.get('categories', {})}, Scores: {input_check.get('category_scores', {})}")

    # Generate response
    response = await client.chat.completions.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    # Check output content
    output_text = response.choices[0].message.content
    output_check = await content_filter.check_content(output_text)

    if not output_check["safe"]:
        raise ValueError(f"Output content filtered: {output_check['reason']}, Categories: {output_check.get('categories', {})}, Scores: {output_check.get('category_scores', {})}")

    return output_text


# Usage example
# content_filter = ContentFilter(client)
# content_filter.add_blocked_terms(["offensive term 1", "offensive term 2"])
# safe_response = await safe_completion("Your prompt here", content_filter)
# print(safe_response)
```

**Reference:**

[Content Filtering Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter)

## 12. Advanced Caching Strategy

**Overview:**

Implement caching strategies to improve performance and reduce costs by storing frequently used responses, This version adds `cached_tokens` information that can be used to measure the caching efficacy.

```python
from functools import lru_cache
import hashlib
import redis
from typing import Optional

class ResponseCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour

    def generate_cache_key(self, prompt: str, model: str) -> str:
        """Generate a unique cache key based on prompt and model."""
        content = f"{prompt}:{model}".encode()
        return hashlib.sha256(content).hexdigest()

    async def get_cached_response(self, prompt: str, model: str) -> Optional[str]:
        cache_key = self.generate_cache_key(prompt, model)
        cached = self.redis_client.get(cache_key)
        return cached.decode() if cached else None

    async def cache_response(self, prompt: str, model: str,
                           response: str, ttl: int = None):
        cache_key = self.generate_cache_key(prompt, model)
        self.redis_client.setex(
            cache_key,
            ttl or self.default_ttl,
            response.encode()
        )

class CachedOpenAIClient:
    def __init__(self, cache: ResponseCache, client: AzureOpenAI):
        self.cache = cache
        self.client = client

    async def get_completion(self, prompt: str,
                           model: str = "gpt-4",
                           use_cache: bool = True) -> str:
        if use_cache:
            cached_response = await self.cache.get_cached_response(prompt, model)
            if cached_response:
                return cached_response

        response = await self.client.chat.completions.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content

        if use_cache:
            await self.cache.cache_response(prompt, model, response_text)

        return response_text

# Usage example
# cache = ResponseCache("redis://localhost:6379")
# cached_client = CachedOpenAIClient(cache, client)
# cached_response = await cached_client.get_completion("Your prompt here")
# print(cached_response)
```

**Reference:**

[Caching Documentation](https://learn.microsoft.com/en-us/azure/architecture/best-practices/caching)
[Prompt caching with Azure OpenAI Service - Azure OpenAI | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/prompt-caching)

## 13. Advanced Integration Patterns

**Overview:**

Explore advanced integration patterns to enhance the functionality and scalability of Azure OpenAI applications. This version expands the integration concept to include Assistants API and more robust content checking and request processing.

```python
import time
from typing import Dict, Any, Optional
from openai import AzureOpenAI


class AzureOpenAIIntegration:
    def __init__(self, config: Dict[str, Any], client: AzureOpenAI):
        self.client = client
        self.cache = ResponseCache(config["redis_url"])
        self.monitor = SystemMonitor()
        self.content_filter = ContentFilter(self.client)

    async def process_request(self,
                            prompt: str,
                            use_cache: bool = True,
                            check_content: bool = True) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # Content filtering
            if check_content:
                content_check = await self.content_filter.check_content(prompt)
                if not content_check["safe"]:
                     raise ValueError(f"Content filtered: {content_check['reason']}, Categories: {content_check.get('categories', {})}, Scores: {content_check.get('category_scores', {})}")

            # Check cache
            if use_cache:
                cached_response = await self.cache.get_cached_response(
                    prompt, "gpt-4"
                )
                if cached_response:
                    return {
                        "response": cached_response,
                        "cached": True,
                        "processing_time": time.time() - start_time
                    }

            # Generate response
            response = await self.client.chat.completions.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content

            # Cache response
            if use_cache:
                await self.cache.cache_response(prompt, "gpt-4", response_text)

            return {
                "response": response_text,
                "cached": False,
                "processing_time": time.time() - start_time,
                "usage": response.usage.dict()
            }
        except Exception as e:
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def process_assistant_request(
            self,
            assistant_id: str,
            thread_id: Optional[str],
            prompt: str,
             use_cache: bool = True,
            check_content: bool = True
    ) -> Dict[str, Any]:
         start_time = time.time()
         try:
              # Content filtering
            if check_content:
                content_check = await self.content_filter.check_content(prompt)
                if not content_check["safe"]:
                     raise ValueError(f"Content filtered: {content_check['reason']}, Categories: {content_check.get('categories', {})}, Scores: {content_check.get('category_scores', {})}")
                
            if not thread_id:
                thread = await client.beta.threads.create()
                thread_id = thread.id
            
            message = await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=prompt
                )
            
            run = await client.beta.threads.runs.create(
                    thread_id = thread_id,
                     assistant_id=assistant_id
                )
                
            while True:
                run_response = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

                if run_response.status == "completed":
                  messages = await client.beta.threads.messages.list(thread_id=thread_id)
                  assistant_message = [message for message in messages.data if message.role == "assistant"]
                  response_text = assistant_message[-1].content[0].text.value if assistant_message and assistant_message[-1].content else None
                  return {
                       "response": response_text,
                       "cached": False,
                       "processing_time": time.time() - start_time,
                        "usage": None
                   }
                if run_response.status in ["failed", "cancelled", "expired"]:
                  return {
                       "error": f"Run failed with status: {run_response.status}",
                       "processing_time": time.time() - start_time
                    }
                await asyncio.sleep(1)
         except Exception as e:
             return {
                 "error": str(e),
                 "processing_time": time.time() - start_time
             }


# Usage example
# integration = AzureOpenAIIntegration(config, client)
# response_data = await integration.process_request("Your prompt here")
# print(response_data)
```

**Reference:**

[Integration Patterns Documentation](https://learn.microsoft.com/en-us/azure/architecture/patterns/)

## 14. Implementing Retrieval-Augmented Generation (RAG)

**Overview:**

Utilize Retrieval-Augmented Generation (RAG) to enhance the quality of AI responses by integrating external knowledge sources. This version also shows how to use Azure Search and the Assistants API, by configuring chat extensions in a thread, and using those in a run to get the responses.

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional


# Setup search client (example)
# search_client = SearchClient(
#     endpoint=os.getenv("SEARCH_ENDPOINT"),
#     index_name="your-index",
#     credential=AzureKeyCredential(os.getenv("SEARCH_KEY"))
# )

async def rag_query(user_query: str, client: AzureOpenAI, search_client: SearchClient):
    # 1. Search relevant documents
    search_results = search_client.search(
        search_text=user_query,
        select=["content", "title"],
        top=3
    )

    # 2. Format context from search results
    context = "\n".join([doc["content"] for doc in search_results])

    # 3. Generate response using context
    messages = [
        {"role": "system", "content": "Use the following context to answer questions:\n" + context},
        {"role": "user", "content": user_query}
    ]

    response = await client.chat.completions.acreate(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content

async def rag_assistant_query(
    user_query: str,
    client: AzureOpenAI,
    assistant_id: str,
    search_index_name: str,
    search_endpoint: str,
    search_key: str,
    embedding_deployment_name: str
    ) -> Dict[str, Any]:
   
    thread = await client.beta.threads.create(
        metadata= { "search_index": search_index_name} ,
        tool_resources = {
             "file_search" : {
                 "vector_stores": [ {
                         "file_ids": [],
                          "metadata":  { "search_index": search_index_name}
                     }
                     ]
             }
         }
    )
    
    
    
    messages = await client.beta.threads.messages.create(
        thread_id = thread.id,
        role = "user",
        content = user_query
    )

    run = await client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = assistant_id,
            tools=[
                {
                      "type": "file_search",
                  }
            ],
            tool_resources={
                "file_search":{
                    "vector_store_ids":[]
                }
            },
            
            
            
            
            
        )
   
    while True:
      run_response = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
      
      if run_response.status == "completed":
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = [message for message in messages.data if message.role == "assistant"]
        response_text = assistant_message[-1].content[0].text.value if assistant_message and assistant_message[-1].content else None
        return {
          "response": response_text,
          "usage": None
        }
      
      if run_response.status in ["failed", "cancelled", "expired"]:
         return {
             "error": f"Run failed with status: {run_response.status}"
          }
      await asyncio.sleep(1)

# Example usage
# query = "What is the capital of France?"
# response = await rag_query(query, client, search_client)
# print(response)

```

**Reference:**

[RAG Documentation](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview)

## 15. Generating Embeddings

**Overview:**

Generate embeddings using Azure OpenAI for tasks like similarity search and clustering, enhancing the ability to analyze and process text data. The `text-embedding-3` model and later versions now support the `dimensions` parameters.

```python
from openai import AzureOpenAI

async def generate_embeddings(text: str, client: AzureOpenAI, model: str ="text-embedding-ada-002", dimensions: Optional[int] = None):
    response = await client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions
    )
    return response.data[0].embedding

# Usage example
# text = "Azure OpenAI provides powerful AI capabilities."
# embedding = await generate_embeddings(text, client)
# print(embedding)

```

**Reference:**

[Generating Embeddings Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/understand-embeddings)

Okay, here are the remaining complete and updated sections (16-19) of the Azure OpenAI Strategy Guide, incorporating the new API knowledge and addressing the necessary changes.

## 16. Azure OpenAI and Sentry Configuration

**Overview:**

Integrate Azure OpenAI with Sentry for error tracking and monitoring, ensuring robust application performance and reliability. This version includes capturing more data from errors, particularly the `inner_error` and `content_filter_results` which are part of the API responses.

**Example Configuration:**

```plaintext
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-10-01-preview

# Sentry Configuration
SENTRY_DSN=https://your-sentry-dsn@sentry.io/your-project-id

# Optional: Azure Cognitive Services for Speech (if used)
# SPEECH_API_KEY=your-speech-api-key
# SPEECH_REGION=eastus2
```

**Client Initialization:**

```python
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import sentry_sdk

# Load environment variables
load_dotenv()

azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-01-preview')
sentry_dsn = os.getenv('SENTRY_DSN')

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

# Initialize Sentry
sentry_sdk.init(
    dsn=sentry_dsn,
    traces_sample_rate=1.0
)
```

**Example Usage with Error Handling:**

```python
try:
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": "What's the weather like today?"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    sentry_sdk.capture_exception(e)
    if hasattr(e, "response") and e.response:
        try:
            error_response = e.response.json()
            sentry_sdk.capture_message(f"API Error Response: {error_response}")
        except:
            sentry_sdk.capture_message("Could not parse error response as JSON")

    print(f"An error occurred: {e}")
```

**Reference:**

* [Azure OpenAI and Sentry Configuration Guide](https://github.com/Azure-Samples/openai/blob/main/Basic_Samples/Functions/working_with_functions.ipynb)
* [Switching between Azure OpenAI and OpenAI endpoints](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints?source=recommendations)

## 17. Stream Response Processing (New Section)

**Overview:**

Handle streaming responses efficiently for real-time content generation. The `stream_options` parameter can be set with `include_usage=True` to include an additional chunk in the stream containing usage information.

```python
from typing import AsyncGenerator, Dict, Any
from openai import AzureOpenAI

async def process_stream_response(prompt: str, client: AzureOpenAI) -> AsyncGenerator[str, None]:
    try:
        stream = await client.chat.completions.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
             stream_options = {"include_usage": True}
        )

        collected_messages = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                collected_messages.append(content)
                yield content
            elif chunk.usage:
                 yield f"Usage: {chunk.usage}"


    except Exception as e:
        yield f"Error during streaming: {str(e)}"


# Usage example with async generator
async def stream_example(client: AzureOpenAI):
    async for content in process_stream_response("Generate a story", client=client):
        print(content, end="", flush=True)

```

## 18. Advanced Configuration Management (New Section)

**Overview:**

Manage advanced configuration options for fine-tuned control over API responses. This version shows how to use all the completion parameters and also the parameters for `max_prompt_tokens` and `max_completion_tokens` that can be used in the Assistants API for cost and context management.

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import AzureOpenAI


@dataclass
class OpenAIConfig:
    temperature: float = 0.7
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        config = {
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed
        }
        if self.max_tokens:
             config["max_tokens"] = self.max_tokens
        if self.top_p:
              config["top_p"] = self.top_p
        return config

    def to_assistant_dict(self) -> Dict[str, Any]:
        config = {
            "temperature": self.temperature,
             "top_p": self.top_p,
             "max_prompt_tokens": self.max_prompt_tokens,
             "max_completion_tokens": self.max_completion_tokens
        }
        return config


class ConfigurableOpenAIClient:
    def __init__(self, config: OpenAIConfig, client: AzureOpenAI):
        self.config = config
        self.client = client

    async def get_completion(
        self,
        prompt: str,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        request_params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            **self.config.to_dict()
        }

        if response_format:
            request_params["response_format"] = response_format

        response = await self.client.chat.completions.acreate(**request_params)
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.dict()
        }
    
    async def get_assistant_run(
        self,
        assistant_id: str,
        thread_id: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
       ) -> Dict[str, Any]:
       
        request_params = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            **self.config.to_assistant_dict()
        }
        if tools:
            request_params["tools"]=tools
        
        
        
        message = await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt
                )
        run = await client.beta.threads.runs.create(**request_params)

        while True:
            run_response = await self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

            if run_response.status == "completed":
              messages = await client.beta.threads.messages.list(thread_id=thread_id)
              assistant_message = [message for message in messages.data if message.role == "assistant"]
              response_text = assistant_message[-1].content[0].text.value if assistant_message and assistant_message[-1].content else None
              return {
                  "content": response_text,
                  "usage": run_response.usage.dict() if run_response.usage else None,
                  "status": run_response.status
                }
            if run_response.status in ["failed", "cancelled", "expired"]:
                 return {
                    "error": f"Run failed with status: {run_response.status}",
                    "usage": None
                  }
            await asyncio.sleep(1)


# Usage example
# config = OpenAIConfig(
#    temperature=0.5,
#    frequency_penalty=0.2,
#   presence_penalty=0.1,
#    seed=42
# )
# configurable_client = ConfigurableOpenAIClient(config, client)

```

## 19. Response Validation and Processing (New Section)

**Overview:**

Implement comprehensive response validation and processing based on API specifications. This version includes validation for all relevant schemas, including those from the Assistants API.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI


class ChatChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None


class ChatResponse(BaseModel):
    id: str
    object: str = Field("chat.completion")
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage
    
class RunCompletionUsage(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  
class RunObject(BaseModel):
  id: str
  object: str = Field("thread.run")
  status: str
  usage: Optional[RunCompletionUsage] = None
  
class ThreadMessageTextContent(BaseModel):
    value: str
    annotations: List[Any]
    
class ThreadMessageContent(BaseModel):
    type: str
    text: Optional[ThreadMessageTextContent] = None

class MessageObject(BaseModel):
  id: str
  object: str = Field("thread.message")
  role: str
  content: List[ThreadMessageContent]
  
  
class ResponseValidator:
    def __init__(self):
        self.validators: Dict[str, Any] = {
            "chat.completion": ChatResponse,
            "thread.run": RunObject
        }

    def validate_response(
        self,
        response: Dict[str, Any],
        response_type: str = "chat.completion"
    ) -> Dict[str, Any]:
        """Validate and process API response"""
        if response_type not in self.validators:
            raise ValueError(f"Unknown response type: {response_type}")

        validator = self.validators[response_type]
        validated = validator(**response)
        return validated.dict()


class ResponseProcessor:
    def __init__(self, validator: ResponseValidator):
        self.validator = validator

    async def process_response(
        self,
        response: Dict[str, Any],
        response_type: str ="chat.completion",
        extract_content: bool = True
    ) -> Dict[str, Any]:
        """Process and validate API response"""
        validated = self.validator.validate_response(response, response_type)

        if extract_content and response_type == "chat.completion":
            return {
                "content": validated["choices"][0]["message"]["content"],
                "usage": validated["usage"],
                "model": validated["model"],
                "finish_reason": validated["choices"][0]["finish_reason"]
            }
        elif extract_content and response_type == "thread.run":
           return {
              "status": validated["status"],
               "usage": validated["usage"] if validated.get("usage") else None
           }


        return validated

# Usage example
validator = ResponseValidator()
processor = ResponseProcessor(validator)

async def make_validated_request(prompt: str, client: AzureOpenAI) -> Dict[str, Any]:
    response = await client.chat.completions.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return await processor.process_response(response.dict())
```

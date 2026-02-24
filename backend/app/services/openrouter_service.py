"""
OpenRouter service as fallback provider for LLM completion.

Provides integration with OpenRouter API for multi-model access
when Anthropic is unavailable or for testing purposes.
"""

import asyncio
import os
from typing import AsyncIterator, Optional, List, Dict, Any
from datetime import datetime

from openai import AsyncOpenAI
from openai.types import Completion

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    StreamingChunk,
    Usage,
    StopReason,
    ThinkingConfig,
)


# Map Anthropic models to OpenRouter equivalents
OPENROUTER_MODEL_MAP = {
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-haiku-4-5": "anthropic/claude-3-5-haiku",
}


class OpenRouterService:
    """
    Service for interacting with OpenRouter API as a fallback provider.

    Supports:
    - Message completion
    - Streaming responses
    - Multiple model providers via OpenRouter
    """

    DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TIMEOUT_MS = 60000

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        """
        Initialize the OpenRouter service.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: API base URL
            timeout_ms: Request timeout in milliseconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Use empty key - service will fail health check
            self.api_key = ""

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout_ms / 1000,
        )
        self.timeout_ms = timeout_ms
        self.base_url = base_url

    def _map_model(self, model: str) -> str:
        """Map Anthropic model to OpenRouter equivalent."""
        return OPENROUTER_MODEL_MAP.get(model, model)

    def _convert_message(self, message: ChatMessage) -> Dict[str, Any]:
        """Convert ChatMessage to OpenAI format."""
        content = message.content if isinstance(message.content, str) else str(message.content)

        return {
            "role": message.role.value,
            "content": content,
        }

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            request: Chat request with messages and config

        Returns:
            ChatResponse with generated content

        Raises:
            ValueError: If API key is not configured
            Exception: If API call fails
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")

        mapped_model = self._map_model(request.model)

        # Build messages
        messages = [self._convert_message(m) for m in request.messages]

        # Make the API call
        response = await self.client.chat.completions.create(
            model=mapped_model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        return ChatResponse(
            id=response.id,
            role=MessageRole.ASSISTANT,
            content=content,
            model=response.model,
            stop_reason=StopReason.END_TURN if choice.finish_reason == "stop" else StopReason.MAX_TOKENS,
            usage=Usage(
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
            provider="openrouter",
        )

    async def chat_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Send a streaming chat completion request.

        Args:
            request: Chat request with stream=True

        Yields:
            StreamingChunk events as they arrive

        Raises:
            ValueError: If API key is not configured
            Exception: If API call fails
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")

        mapped_model = self._map_model(request.model)
        messages = [self._convert_message(m) for m in request.messages]

        stream = await self.client.chat.completions.create(
            model=mapped_model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True,
        )

        response_id = "openrouter-" + datetime.utcnow().isoformat()

        yield StreamingChunk(
            id=response_id,
            event_type="message_start",
        )

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    yield StreamingChunk(
                        id=response_id,
                        event_type="content_block_delta",
                        content=delta.content,
                    )

            if chunk.usage:
                yield StreamingChunk(
                    id=response_id,
                    event_type="message_delta",
                    usage=Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    ),
                    finish_reason=StopReason.END_TURN if chunk.choices and chunk.choices[0].finish_reason == "stop" else None,
                )

        yield StreamingChunk(
            id=response_id,
            event_type="message_stop",
        )

    async def health_check(self) -> bool:
        """
        Check if the OpenRouter service is healthy.

        Returns:
            True if service is available, False otherwise
        """
        if not self.api_key:
            return False

        try:
            await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="openai/gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                ),
                timeout=5.0,
            )
            return True
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """
        List available models via OpenRouter.

        Returns:
            List of available model identifiers
        """
        return list(OPENROUTER_MODEL_MAP.values())

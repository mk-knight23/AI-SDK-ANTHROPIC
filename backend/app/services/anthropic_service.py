"""
Anthropic Claude service for chat, streaming, and extended thinking.

Provides integration with Anthropic's Claude API including:
- Message completion with extended thinking
- Streaming responses
- Tool/function calling
- Computer use capabilities
"""

import asyncio
import os
from typing import AsyncIterator, Optional, List, Dict, Any, Union
from datetime import datetime

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolUseBlock,
)

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    StreamingChunk,
    Usage,
    StopReason,
    ThinkingConfig,
    ContentBlock,
    ToolUse,
    ToolResult,
)


class AnthropicService:
    """
    Service for interacting with Anthropic's Claude API.

    Supports:
    - Standard message completion
    - Extended thinking (chain-of-thought reasoning)
    - Server-sent events streaming
    - Tool/function calling
    - Computer use
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TIMEOUT_MS = 60000

    # Models that support extended thinking
    THINKING_ENABLED_MODELS = {
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }

    # Models that support computer use
    COMPUTER_USE_MODELS = {
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        """
        Initialize the Anthropic service.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Optional custom base URL
            timeout_ms: Request timeout in milliseconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout_ms / 1000,  # Convert to seconds
        )
        self.timeout_ms = timeout_ms

    def _supports_thinking(self, model: str) -> bool:
        """Check if model supports extended thinking."""
        return model in self.THINKING_ENABLED_MODELS

    def _supports_computer_use(self, model: str) -> bool:
        """Check if model supports computer use."""
        return model in self.COMPUTER_USE_MODELS

    def _convert_message_to_param(self, message: ChatMessage) -> MessageParam:
        """
        Convert ChatMessage to Anthropic MessageParam format.

        Args:
            message: ChatMessage to convert

        Returns:
            MessageParam for Anthropic API
        """
        role = message.role.value

        if isinstance(message.content, str):
            return {"role": role, "content": message.content}

        # Handle content blocks
        blocks = []
        for block in message.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": block.tool_use.id,
                    "name": block.tool_use.name,
                    "input": block.tool_use.input,
                })
            elif block.type == "tool_result":
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.tool_result.tool_use_id,
                    "content": block.tool_result.content,
                    "is_error": block.tool_result.is_error,
                })
            elif block.type == "image":
                blocks.append({
                    "type": "image",
                    "source": block.source,
                })

        return {"role": role, "content": blocks}  # type: ignore

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
            ValueError: If request is invalid
            anthropic.APIError: If API call fails
        """
        thinking_config = request.thinking or ThinkingConfig()
        use_thinking = thinking_config.enabled and self._supports_thinking(request.model)

        # Build API parameters
        api_params: Dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "messages": [self._convert_message_to_param(m) for m in request.messages],
        }

        # Add system prompt if provided
        if request.system:
            api_params["system"] = request.system

        # Add extended thinking if enabled
        if use_thinking:
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_config.budget_tokens or thinking_config.max_tokens,
            }

        # Add tools if provided
        if request.tools:
            api_params["tools"] = request.tools

        # Make the API call
        response: Message = await self.client.messages.create(**api_params)

        # Extract content and thinking
        content_text = ""
        thinking_content = None
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "tool_use":
                tool_uses.append(ToolUse(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        # Build response
        return ChatResponse(
            id=response.id,
            role=MessageRole.ASSISTANT,
            content=content_text,
            model=response.model,
            stop_reason=StopReason(response.stop_reason) if response.stop_reason else None,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            thinking=thinking_content,
            tool_use=tool_uses if tool_uses else None,
            provider="anthropic",
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
            ValueError: If request is invalid
            anthropic.APIError: If API call fails
        """
        thinking_config = request.thinking or ThinkingConfig()
        use_thinking = thinking_config.enabled and self._supports_thinking(request.model)

        # Build API parameters
        api_params: Dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "messages": [self._convert_message_to_param(m) for m in request.messages],
            "stream": True,
        }

        if request.system:
            api_params["system"] = request.system

        if use_thinking:
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_config.budget_tokens or thinking_config.max_tokens,
            }

        if request.tools:
            api_params["tools"] = request.tools

        # Stream the response
        stream = await self.client.messages.create(**api_params)

        response_id = None
        content_index = 0
        accumulated_content = ""

        async for event in stream:
            # Get response ID from first event
            if response_id is None and hasattr(event, "message"):
                response_id = event.message.id

            # Handle different event types
            event_type = event.type

            if event_type == "message_start":
                yield StreamingChunk(
                    id=response_id or "unknown",
                    event_type="message_start",
                )

            elif event_type == "message_delta":
                usage = Usage(
                    input_tokens=event.usage.input_tokens if hasattr(event, "usage") else 0,
                    output_tokens=event.usage.output_tokens if hasattr(event, "usage") else 0,
                    total_tokens=0,
                )
                if hasattr(event, "usage"):
                    usage.total_tokens = usage.input_tokens + usage.output_tokens

                yield StreamingChunk(
                    id=response_id or "unknown",
                    event_type="message_delta",
                    usage=usage,
                    finish_reason=StopReason(event.delta.stop_reason) if event.delta.stop_reason else None,
                )

            elif event_type == "message_stop":
                yield StreamingChunk(
                    id=response_id or "unknown",
                    event_type="message_stop",
                )

            elif event_type == "content_block_start":
                if event.content_block.type == "thinking":
                    yield StreamingChunk(
                        id=response_id or "unknown",
                        event_type="content_block_start",
                        index=event.index,
                        thinking=True,
                    )
                else:
                    yield StreamingChunk(
                        id=response_id or "unknown",
                        event_type="content_block_start",
                        index=event.index,
                    )

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    accumulated_content += delta.text
                    yield StreamingChunk(
                        id=response_id or "unknown",
                        event_type="content_block_delta",
                        content=delta.text,
                        index=event.index,
                    )
                elif delta.type == "thinking_delta":
                    yield StreamingChunk(
                        id=response_id or "unknown",
                        event_type="content_block_delta",
                        content=delta.thinking,
                        index=event.index,
                        thinking=True,
                    )

            elif event_type == "content_block_stop":
                yield StreamingChunk(
                    id=response_id or "unknown",
                    event_type="content_block_stop",
                    index=event.index,
                )

    async def health_check(self) -> bool:
        """
        Check if the Anthropic service is healthy.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Make a minimal API call
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model="claude-3-haiku-4-5",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}],
                ),
                timeout=5.0,
            )
            return True
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """
        List available Claude models.

        Returns:
            List of model identifiers
        """
        return [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ]

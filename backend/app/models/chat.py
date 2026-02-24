"""Chat and message models."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Message role types."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ToolUse(BaseModel):
    """Tool use block in a message."""

    id: str = Field(..., description="Unique tool use identifier")
    name: str = Field(..., description="Tool name")
    input: Dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")


class ToolResult(BaseModel):
    """Tool result block in a message."""

    tool_use_id: str = Field(..., description="Corresponding tool use ID")
    content: str = Field(..., description="Tool result content")
    is_error: bool = Field(default=False, description="Whether the tool execution failed")


class ContentBlock(BaseModel):
    """Content block in a message."""

    type: Literal["text", "tool_use", "tool_result", "image"] = Field(
        ..., description="Content block type"
    )
    text: Optional[str] = Field(None, description="Text content")
    tool_use: Optional[ToolUse] = Field(None, description="Tool use block")
    tool_result: Optional[ToolResult] = Field(None, description="Tool result block")
    source: Optional[Dict[str, Any]] = Field(None, description="Image source data")


class ChatMessage(BaseModel):
    """Chat message with support for various content types."""

    role: MessageRole = Field(..., description="Message role")
    content: Union[str, List[ContentBlock]] = Field(
        ..., description="Message content (string or list of content blocks)"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow, description="Message timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional metadata"
    )


class ThinkingConfig(BaseModel):
    """Extended thinking configuration."""

    enabled: bool = Field(default=True, description="Enable extended thinking")
    max_tokens: int = Field(
        default=20000, ge=1000, le=100000, description="Max thinking tokens"
    )
    budget_tokens: Optional[int] = Field(
        default=None, ge=1000, le=100000, description="Budget for thinking phase"
    )


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    use_fallback: bool = Field(default=False, description="Use fallback provider")
    fallback_provider: Literal["openrouter"] = Field(
        default="openrouter", description="Fallback provider name"
    )
    timeout_ms: int = Field(default=60000, ge=5000, description="Request timeout")


class ChatRequest(BaseModel):
    """Chat completion request."""

    messages: List[ChatMessage] = Field(..., min_items=1, description="Conversation messages")
    model: str = Field(
        default="claude-sonnet-4-6", description="Anthropic model to use"
    )
    max_tokens: int = Field(
        default=4096, ge=1, le=8192, description="Max completion tokens"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Sampling temperature"
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stream: bool = Field(default=False, description="Enable streaming response")
    thinking: Optional[ThinkingConfig] = Field(
        default=None, description="Extended thinking config"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Available tools for function calling"
    )
    provider: Optional[ProviderConfig] = Field(
        default=None, description="Provider configuration"
    )
    system: Optional[str] = Field(None, description="System prompt")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ],
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "stream": False,
            }
        }


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int = Field(..., description="Input prompt tokens")
    output_tokens: int = Field(..., description="Output completion tokens")
    total_tokens: int = Field(..., description="Total tokens used")
    thinking_tokens: Optional[int] = Field(None, description="Thinking phase tokens")


class StopReason(str, Enum):
    """Reason for completion termination."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class ChatResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(..., description="Response ID")
    role: MessageRole = Field(MessageRole.ASSISTANT, description="Response role")
    content: str = Field(..., description="Response text content")
    model: str = Field(..., description="Model used")
    stop_reason: Optional[StopReason] = Field(None, description="Stop reason")
    usage: Usage = Field(..., description="Token usage")
    thinking: Optional[str] = Field(None, description="Extended thinking content")
    tool_use: Optional[List[ToolUse]] = Field(None, description="Tool use blocks")
    provider: str = Field(default="anthropic", description="Provider used")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class StreamingChunk(BaseModel):
    """Streaming response chunk."""

    id: str = Field(..., description="Chunk/event ID")
    event_type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "error",
    ] = Field(..., description="Event type")
    content: Optional[str] = Field(None, description="Content delta")
    thinking: Optional[bool] = Field(None, description="Is thinking content")
    index: Optional[int] = Field(None, description="Content block index")
    usage: Optional[Usage] = Field(None, description="Cumulative token usage")
    error: Optional[str] = Field(None, description="Error message if event_type is error")
    finish_reason: Optional[StopReason] = Field(None, description="Completion reason")

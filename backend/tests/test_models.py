"""Tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    ThinkingConfig,
    Usage,
    StreamingChunk,
    ContentBlock,
    ToolUse,
)
from app.models.artifact import (
    ArtifactCreateRequest,
    ArtifactResponse,
    ArtifactType,
    ArtifactStatus,
)
from app.models.computer_use import (
    ComputerUseRequest,
    ComputerAction,
    ActionType,
)
from app.models.health import HealthResponse
from app.models.provider import ProviderStatus, ProviderSwitchRequest


class TestChatModels:
    """Tests for chat-related models."""

    def test_chat_message_basic(self):
        """Test basic chat message creation."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Hello, world!"
        )
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.timestamp is not None

    def test_chat_message_with_content_blocks(self):
        """Test chat message with content blocks."""
        tool_use = ToolUse(id="tool_123", name="search", input={"query": "test"})
        block = ContentBlock(type="tool_use", tool_use=tool_use)

        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=[block]
        )
        assert message.role == MessageRole.ASSISTANT
        assert isinstance(message.content, list)
        assert len(message.content) == 1

    def test_chat_request_validation(self):
        """Test chat request validation."""
        request = ChatRequest(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Test")
            ],
            model="claude-sonnet-4-6",
            max_tokens=100,
        )
        assert len(request.messages) == 1
        assert request.model == "claude-sonnet-4-6"
        assert request.max_tokens == 100

    def test_chat_request_invalid_temperature(self):
        """Test chat request rejects invalid temperature."""
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Test")],
                temperature=2.0,  # Invalid: > 1.0
            )

    def test_thinking_config(self):
        """Test thinking configuration."""
        config = ThinkingConfig(
            enabled=True,
            max_tokens=10000,
        )
        assert config.enabled is True
        assert config.max_tokens == 10000

    def test_thinking_config_invalid_max_tokens(self):
        """Test thinking config rejects invalid max_tokens."""
        with pytest.raises(ValidationError):
            ThinkingConfig(max_tokens=200000)  # Too large

    def test_usage_model(self):
        """Test usage model."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_chat_response(self):
        """Test chat response model."""
        response = ChatResponse(
            id="msg_123",
            role=MessageRole.ASSISTANT,
            content="Hello!",
            model="claude-sonnet-4-6",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert response.id == "msg_123"
        assert response.content == "Hello!"
        assert response.provider == "anthropic"

    def test_streaming_chunk(self):
        """Test streaming chunk model."""
        chunk = StreamingChunk(
            id="chunk_123",
            event_type="content_block_delta",
            content="Hello",
        )
        assert chunk.id == "chunk_123"
        assert chunk.content == "Hello"


class TestArtifactModels:
    """Tests for artifact-related models."""

    def test_artifact_create_request(self):
        """Test artifact creation request."""
        request = ArtifactCreateRequest(
            content="# Test",
            artifact_type=ArtifactType.PDF,
            title="Test PDF",
        )
        assert request.artifact_type == ArtifactType.PDF
        assert request.title == "Test PDF"

    def test_artifact_response(self):
        """Test artifact response."""
        response = ArtifactResponse(
            id="art_123",
            artifact_type=ArtifactType.PDF,
            status=ArtifactStatus.COMPLETED,
            filename="test.pdf",
            download_url="/api/artifacts/art_123/download",
        )
        assert response.id == "art_123"
        assert response.status == ArtifactStatus.COMPLETED

    def test_artifact_types(self):
        """Test all artifact types are valid."""
        assert ArtifactType.PDF == "pdf"
        assert ArtifactType.CODE == "code"
        assert ArtifactType.MARKDOWN == "markdown"
        assert ArtifactType.HTML == "html"
        assert ArtifactType.TEXT == "text"


class TestComputerUseModels:
    """Tests for computer use models."""

    def test_computer_action_screenshot(self):
        """Test screenshot action."""
        action = ComputerAction(type=ActionType.SCREENSHOT)
        assert action.type == ActionType.SCREENSHOT

    def test_computer_action_click(self):
        """Test click action."""
        action = ComputerAction(
            type=ActionType.CLICK,
            coordinates={"x": 100, "y": 200},
            button="left",
        )
        assert action.type == ActionType.CLICK
        assert action.coordinates == {"x": 100, "y": 200}

    def test_computer_use_request(self):
        """Test computer use request."""
        request = ComputerUseRequest(
            actions=[
                ComputerAction(type=ActionType.SCREENSHOT),
            ],
            headless=True,
        )
        assert len(request.actions) == 1
        assert request.headless is True


class TestHealthModels:
    """Tests for health-related models."""

    def test_health_response(self):
        """Test health response."""
        response = HealthResponse(
            status="healthy",
            timestamp="2025-02-23T10:00:00Z",
            version="1.0.0",
            service="anthropic-api",
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"


class TestProviderModels:
    """Tests for provider-related models."""

    def test_provider_status(self):
        """Test provider status."""
        status = ProviderStatus(
            name="anthropic",
            available=True,
            latency_ms=100,
            models=["claude-sonnet-4-6"],
        )
        assert status.name == "anthropic"
        assert status.available is True

    def test_provider_switch_request(self):
        """Test provider switch request."""
        request = ProviderSwitchRequest(
            provider="openrouter",
            force=True,
        )
        assert request.provider == "openrouter"
        assert request.force is True

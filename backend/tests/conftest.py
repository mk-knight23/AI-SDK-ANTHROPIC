"""Test configuration and fixtures."""

import os
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
import pytest

from fastapi.testclient import TestClient

from main import app
from app.services import (
    AnthropicService,
    OpenRouterService,
    LLMRouter,
    ArtifactService,
    ComputerUseService,
)


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_anthropic_service(mock_anthropic_client):
    """Mock Anthropic service."""
    service = MagicMock(spec=AnthropicService)
    service.chat = AsyncMock()
    service.chat_stream = AsyncMock()
    service.health_check = AsyncMock(return_value=True)
    service.list_models = AsyncMock(return_value=["claude-opus-4-6", "claude-sonnet-4-6"])
    return service


@pytest.fixture
def mock_openrouter_service(mock_openrouter_client):
    """Mock OpenRouter service."""
    service = MagicMock(spec=OpenRouterService)
    service.chat = AsyncMock()
    service.chat_stream = AsyncMock()
    service.health_check = AsyncMock(return_value=False)  # Default unavailable
    service.list_models = AsyncMock(return_value=["anthropic/claude-sonnet-4-6"])
    return service


@pytest.fixture
def mock_llm_router(mock_anthropic_service, mock_openrouter_service):
    """Mock LLM router."""
    router = MagicMock(spec=LLMRouter)
    router.chat = AsyncMock()
    router.chat_stream = AsyncMock()
    router.get_provider_status = AsyncMock(return_value=[])
    router.health_check = AsyncMock(return_value={"anthropic": True, "openrouter": False})
    return router


@pytest.fixture
def artifact_service(tmp_path):
    """Create artifact service with temp directory."""
    original_storage = ArtifactService.STORAGE_DIR
    ArtifactService.STORAGE_DIR = str(tmp_path / "artifacts")
    service = ArtifactService()
    yield service
    ArtifactService.STORAGE_DIR = original_storage


@pytest.fixture
def computer_use_service():
    """Create computer use service."""
    return ComputerUseService()


@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_chat_request():
    """Sample chat request."""
    from app.models.chat import ChatRequest, ChatMessage, MessageRole

    return ChatRequest(
        messages=[
            ChatMessage(role=MessageRole.USER, content="Hello, how are you?")
        ],
        model="claude-sonnet-4-6",
        max_tokens=100,
    )


@pytest.fixture
def sample_artifact_request():
    """Sample artifact request."""
    from app.models.artifact import ArtifactCreateRequest, ArtifactType

    return ArtifactCreateRequest(
        content="# Hello World\n\nThis is a test document.",
        artifact_type=ArtifactType.PDF,
        title="Test PDF",
    )


@pytest.fixture
def sample_computer_use_request():
    """Sample computer use request."""
    from app.models.computer_use import ComputerUseRequest, ComputerAction, ActionType

    return ComputerUseRequest(
        actions=[
            ComputerAction(type=ActionType.SCREENSHOT),
        ],
        headless=True,
    )

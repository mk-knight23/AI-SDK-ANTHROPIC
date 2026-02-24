"""Tests for service layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.anthropic_service import AnthropicService
from app.services.openrouter_service import OpenRouterService
from app.services.llm_router import LLMRouter, ProviderHealth
from app.services.artifact_service import ArtifactService
from app.services.computer_use_service import ComputerUseService, BrowserSession

from app.models.chat import (
    ChatRequest,
    ChatMessage,
    MessageRole,
    ChatResponse,
    Usage,
    StopReason,
    ThinkingConfig,
)
from app.models.artifact import (
    ArtifactCreateRequest,
    ArtifactResponse,
    ArtifactType,
    ArtifactStatus,
)


@pytest.mark.services
class TestAnthropicService:
    """Tests for Anthropic service."""

    @pytest.fixture
    def service(self):
        """Create service with mocked client."""
        with patch("app.services.anthropic_service.AsyncAnthropic"):
            return AnthropicService(api_key="test_key")

    def test_init_requires_api_key(self):
        """Test service requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError):
                AnthropicService()

    def test_supports_thinking_for_valid_models(self, service):
        """Test thinking support detection."""
        assert service._supports_thinking("claude-sonnet-4-6") is True
        assert service._supports_thinking("claude-opus-4-6") is True
        assert service._supports_thinking("unknown-model") is False

    def test_supports_computer_use_for_valid_models(self, service):
        """Test computer use support detection."""
        assert service._supports_computer_use("claude-sonnet-4-6") is True
        assert service._supports_computer_use("claude-opus-4-6") is True
        assert service._supports_computer_use("unknown-model") is False

    @pytest.mark.asyncio
    async def test_chat_success(self, service):
        """Test successful chat completion."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.id = "msg_123"
        mock_response.model = "claude-sonnet-4-6"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.content = []

        service.client.messages.create = AsyncMock(return_value=mock_response)

        request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            model="claude-sonnet-4-6",
        )

        response = await service.chat(request)

        assert response.id == "msg_123"
        assert response.model == "claude-sonnet-4-6"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check."""
        service.client.messages.create = AsyncMock(return_value=MagicMock())

        result = await service.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, service):
        """Test health check on failure."""
        service.client.messages.create = AsyncMock(side_effect=Exception("API error"))

        result = await service.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_models(self, service):
        """Test listing available models."""
        models = await service.list_models()

        assert "claude-opus-4-6" in models
        assert "claude-sonnet-4-6" in models
        assert "claude-haiku-4-5" in models


@pytest.mark.services
class TestOpenRouterService:
    """Tests for OpenRouter service."""

    @pytest.fixture
    def service(self):
        """Create service with mocked client."""
        with patch("app.services.openrouter_service.AsyncOpenAI"):
            return OpenRouterService(api_key="test_key")

    def test_init_without_api_key(self):
        """Test service can initialize without API key."""
        with patch.dict("os.environ", {}, clear=True):
            service = OpenRouterService()
            assert service.api_key == ""

    def test_map_model(self, service):
        """Test model mapping."""
        assert service._map_model("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"
        assert service._map_model("unknown-model") == "unknown-model"

    @pytest.mark.asyncio
    async def test_chat_without_api_key(self, service):
        """Test chat fails without API key."""
        service.api_key = ""

        request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        )

        with pytest.raises(ValueError, match="API key not configured"):
            await service.chat(request)


@pytest.mark.services
class TestLLMRouter:
    """Tests for LLM router."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic service."""
        service = MagicMock()
        service.chat = AsyncMock()
        service.health_check = AsyncMock(return_value=True)
        service.list_models = AsyncMock(return_value=["claude-sonnet-4-6"])
        return service

    @pytest.fixture
    def mock_openrouter(self):
        """Mock OpenRouter service."""
        service = MagicMock()
        service.chat = AsyncMock()
        service.health_check = AsyncMock(return_value=False)
        service.list_models = AsyncMock(return_value=["anthropic/claude-sonnet-4-6"])
        return service

    @pytest.fixture
    def router(self, mock_anthropic, mock_openrouter):
        """Create router with mocked services."""
        with patch("app.services.llm_router.AnthropicService", return_value=mock_anthropic), \
             patch("app.services.llm_router.OpenRouterService", return_value=mock_openrouter):
            return LLMRouter()

    def test_provider_health_tracking(self):
        """Test provider health tracking."""
        health = ProviderHealth("test")

        assert health.available is True
        assert health.failure_count == 0

        health.record_success(100)
        assert health.available is True
        assert health.failure_count == 0
        assert health.get_average_latency() == 100

        health.record_failure("Test error")
        assert health.failure_count == 1

        # After 3 failures, should be unavailable
        health.record_failure("Test error")
        health.record_failure("Test error")
        assert health.available is False

    @pytest.mark.asyncio
    async def test_chat_uses_primary_provider(self, router, mock_anthropic):
        """Test chat uses primary (Anthropic) provider."""
        mock_response = ChatResponse(
            id="msg_123",
            role=MessageRole.ASSISTANT,
            content="Hello!",
            model="claude-sonnet-4-6",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_anthropic.chat.return_value = mock_response

        request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        )

        response = await router.chat(request)

        assert response.content == "Hello!"
        mock_anthropic.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, router):
        """Test health check updates provider status."""
        results = await router.health_check()

        assert "anthropic" in results
        assert "openrouter" in results

    @pytest.mark.asyncio
    async def test_get_provider_status(self, router):
        """Test getting provider status."""
        statuses = await router.get_provider_status()

        assert len(statuses) == 2
        assert statuses[0].name in ["anthropic", "openrouter"]

    def test_force_provider_switch(self, router):
        """Test forcing provider switch."""
        router.force_provider_switch("openrouter")
        assert router.force_provider == "openrouter"

        router.clear_force_provider()
        assert router.force_provider is None


@pytest.mark.services
class TestArtifactService:
    """Tests for artifact service."""

    def test_generate_filename(self, artifact_service):
        """Test filename generation."""
        filename = artifact_service._generate_filename(
            ArtifactType.PDF,
            "Test Document",
            None,
        )
        assert "test_document" in filename
        assert ".pdf" in filename

    def test_sanitize_filename(self, artifact_service):
        """Test filename sanitization."""
        dirty = "test<>:\"/\\|?*file.txt"
        clean = artifact_service._sanitize_filename(dirty)
        assert "<" not in clean
        assert ">" not in clean
        assert ":" not in clean

    @pytest.mark.asyncio
    async def test_create_pdf_artifact(self, artifact_service):
        """Test PDF artifact creation."""
        request = ArtifactCreateRequest(
            content="# Test Document\n\nSome content.",
            artifact_type=ArtifactType.PDF,
            title="Test PDF",
        )

        response = await artifact_service.create_artifact(request)

        assert response.status == ArtifactStatus.COMPLETED
        assert response.artifact_type == ArtifactType.PDF
        assert response.filename.endswith(".pdf")
        assert response.file_size_bytes > 0

    @pytest.mark.asyncio
    async def test_create_markdown_artifact(self, artifact_service):
        """Test markdown artifact creation."""
        request = ArtifactCreateRequest(
            content="# Hello World",
            artifact_type=ArtifactType.MARKDOWN,
        )

        response = await artifact_service.create_artifact(request)

        assert response.status == ArtifactStatus.COMPLETED
        assert response.artifact_type == ArtifactType.MARKDOWN
        assert response.filename.endswith(".md")

    @pytest.mark.asyncio
    async def test_create_html_artifact(self, artifact_service):
        """Test HTML artifact creation."""
        request = ArtifactCreateRequest(
            content="<p>Hello World</p>",
            artifact_type=ArtifactType.HTML,
        )

        response = await artifact_service.create_artifact(request)

        assert response.status == ArtifactStatus.COMPLETED
        assert response.artifact_type == ArtifactType.HTML
        assert response.filename.endswith(".html")

    @pytest.mark.asyncio
    async def test_create_text_artifact(self, artifact_service):
        """Test text artifact creation."""
        request = ArtifactCreateRequest(
            content="Plain text content",
            artifact_type=ArtifactType.TEXT,
        )

        response = await artifact_service.create_artifact(request)

        assert response.status == ArtifactStatus.COMPLETED
        assert response.artifact_type == ArtifactType.TEXT
        assert response.filename.endswith(".txt")

    @pytest.mark.asyncio
    async def test_create_code_artifact(self, artifact_service):
        """Test code artifact creation."""
        request = ArtifactCreateRequest(
            content="def hello():\n    print('Hello')",
            artifact_type=ArtifactType.CODE,
        )

        response = await artifact_service.create_artifact(request)

        assert response.status == ArtifactStatus.COMPLETED
        assert response.artifact_type == ArtifactType.CODE

    @pytest.mark.asyncio
    async def test_list_artifacts(self, artifact_service):
        """Test listing artifacts."""
        # Create a test artifact
        request = ArtifactCreateRequest(
            content="Test content",
            artifact_type=ArtifactType.TEXT,
        )
        await artifact_service.create_artifact(request)

        # List all artifacts
        artifacts = await artifact_service.list_artifacts()

        assert len(artifacts) > 0

    @pytest.mark.asyncio
    async def test_list_artifacts_by_type(self, artifact_service):
        """Test listing artifacts filtered by type."""
        # Create test artifacts
        await artifact_service.create_artifact(
            ArtifactCreateRequest(content="Test", artifact_type=ArtifactType.TEXT),
        )
        await artifact_service.create_artifact(
            ArtifactCreateRequest(content="# Test", artifact_type=ArtifactType.PDF),
        )

        # List only PDF artifacts
        pdf_artifacts = await artifact_service.list_artifacts(ArtifactType.PDF)

        for artifact in pdf_artifacts:
            assert artifact.artifact_type == ArtifactType.PDF

    @pytest.mark.asyncio
    async def test_delete_artifact(self, artifact_service):
        """Test deleting an artifact."""
        # Create an artifact
        request = ArtifactCreateRequest(
            content="Test content",
            artifact_type=ArtifactType.TEXT,
        )
        artifact = await artifact_service.create_artifact(request)

        # Delete it
        success = await artifact_service.delete_artifact(artifact.id)
        assert success is True

        # Verify it's gone
        result = await artifact_service.get_artifact(artifact.id)
        assert result is None


@pytest.mark.services
class TestComputerUseService:
    """Tests for computer use service."""

    def test_browser_session(self):
        """Test browser session creation."""
        session = BrowserSession("session_123", headless=True)

        assert session.session_id == "session_123"
        assert session.headless is True
        assert session.actions_performed == 0

        session.record_action()
        assert session.actions_performed == 1

    @pytest.mark.asyncio
    async def test_execute_screenshot_action(self):
        """Test executing screenshot action."""
        service = ComputerUseService()

        from app.models.computer_use import ComputerAction, ActionType

        action = ComputerAction(type=ActionType.SCREENSHOT)
        request = MagicMock()
        request.actions = [action]
        request.session_id = None
        request.headless = True

        response = await service.execute_actions(request)

        assert response.session_id is not None
        assert len(response.results) == 1
        assert response.results[0].success is True
        assert response.results[0].screenshot_data is not None

    @pytest.mark.asyncio
    async def test_execute_click_action(self):
        """Test executing click action."""
        service = ComputerUseService()

        from app.models.computer_use import ComputerAction, ActionType

        action = ComputerAction(
            type=ActionType.CLICK,
            coordinates={"x": 100, "y": 200},
        )
        request = MagicMock()
        request.actions = [action]

        response = await service.execute_actions(request)

        assert response.results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_navigate_action(self):
        """Test executing navigate action."""
        service = ComputerUseService()

        from app.models.computer_use import ComputerAction, ActionType

        action = ComputerAction(
            type=ActionType.NAVIGATE,
            url="https://example.com",
        )
        request = MagicMock()
        request.actions = [action]

        response = await service.execute_actions(request)

        assert response.results[0].success is True

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session creation and cleanup."""
        service = ComputerUseService()

        from app.models.computer_use import ComputerAction, ActionType

        # Create session via action
        action = ComputerAction(type=ActionType.SCREENSHOT)
        request = MagicMock()
        request.actions = [action]
        request.session_id = None

        response = await service.execute_actions(request)
        session_id = response.session_id

        # List sessions
        sessions = await service.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id

        # Close session
        success = await service.close_session(session_id)
        assert success is True

        # Verify closed
        sessions = await service.list_sessions()
        assert len(sessions) == 0

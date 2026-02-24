"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from main import app
from app.models.chat import ChatMessage, MessageRole, Usage


@pytest.mark.api
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        with patch("main.llm_router") as mock_router:
            mock_router.health_check = AsyncMock(return_value={"anthropic": True})

            response = test_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data


@pytest.mark.api
class TestProviderEndpoints:
    """Tests for provider management endpoints."""

    def test_get_providers(self, test_client):
        """Test getting provider status."""
        with patch("main.llm_router") as mock_router:
            mock_router.get_provider_status = AsyncMock(return_value=[])

            response = test_client.get("/api/providers")
            assert response.status_code == 200
            assert isinstance(response.json(), list)

    def test_switch_provider(self, test_client):
        """Test switching provider."""
        with patch("main.llm_router") as mock_router:
            mock_router.set_preferred_provider = MagicMock()

            response = test_client.post(
                "/api/providers/switch",
                json={"provider": "openrouter", "force": False},
            )
            assert response.status_code == 200

    def test_switch_provider_invalid(self, test_client):
        """Test switching to invalid provider."""
        with patch("main.llm_router") as mock_router:
            mock_router.set_preferred_provider = MagicMock(
                side_effect=ValueError("Unknown provider")
            )

            response = test_client.post(
                "/api/providers/switch",
                json={"provider": "invalid", "force": False},
            )
            assert response.status_code == 400

    def test_providers_health_check(self, test_client):
        """Test providers health check endpoint."""
        with patch("main.llm_router") as mock_router:
            mock_router.health_check = AsyncMock(
                return_value={"anthropic": True, "openrouter": False}
            )

            response = test_client.post("/api/providers/health")
            assert response.status_code == 200
            data = response.json()
            assert "providers" in data


@pytest.mark.api
class TestChatEndpoints:
    """Tests for chat endpoints."""

    def test_chat_completion(self, test_client, sample_chat_request):
        """Test chat completion endpoint."""
        with patch("main.llm_router") as mock_router:
            from app.models.chat import ChatResponse

            mock_response = ChatResponse(
                id="msg_123",
                role=MessageRole.ASSISTANT,
                content="Hello! How can I help you?",
                model="claude-sonnet-4-6",
                usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            )
            mock_router.chat = AsyncMock(return_value=mock_response)

            response = test_client.post(
                "/api/chat",
                json=sample_chat_request.model_dump(),
            )
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Hello! How can I help you?"
            assert data["id"] == "msg_123"

    def test_chat_completion_with_thinking(self, test_client):
        """Test chat completion with extended thinking."""
        with patch("main.llm_router") as mock_router:
            from app.models.chat import ChatResponse, ThinkingConfig

            mock_response = ChatResponse(
                id="msg_456",
                role=MessageRole.ASSISTANT,
                content="Here's the answer.",
                model="claude-sonnet-4-6",
                usage=Usage(input_tokens=50, output_tokens=100, total_tokens=150),
                thinking="Let me think about this step by step...",
            )
            mock_router.chat = AsyncMock(return_value=mock_response)

            request_data = {
                "messages": [
                    {"role": "user", "content": "Solve this complex problem"}
                ],
                "thinking": {"enabled": True, "max_tokens": 10000},
            }

            response = test_client.post("/api/chat", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["thinking"] is not None

    def test_chat_stream(self, test_client):
        """Test chat streaming endpoint."""
        with patch("main.llm_router") as mock_router:
            from app.models.chat import StreamingChunk

            async def mock_stream():
                yield StreamingChunk(
                    id="msg_789",
                    event_type="content_block_delta",
                    content="Hello",
                )
                yield StreamingChunk(
                    id="msg_789",
                    event_type="content_block_delta",
                    content=" world",
                )

            mock_router.chat_stream = mock_stream

            request_data = {
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "stream": True,
            }

            response = test_client.post("/api/chat/stream", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_completion_error(self, test_client):
        """Test chat completion with error."""
        with patch("main.llm_router") as mock_router:
            mock_router.chat = AsyncMock(side_effect=Exception("API error"))

            response = test_client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "Test"}]},
            )
            assert response.status_code == 500


@pytest.mark.api
class TestArtifactEndpoints:
    """Tests for artifact endpoints."""

    def test_create_artifact(self, test_client, sample_artifact_request):
        """Test creating an artifact."""
        with patch("main.artifact_service") as mock_service:
            from app.models.artifact import ArtifactResponse, ArtifactStatus

            mock_response = ArtifactResponse(
                id="art_123",
                artifact_type=sample_artifact_request.artifact_type,
                status=ArtifactStatus.COMPLETED,
                filename="test_20250223.pdf",
                download_url="/api/artifacts/art_123/download",
                file_size_bytes=1234,
            )
            mock_service.create_artifact = AsyncMock(return_value=mock_response)

            response = test_client.post(
                "/api/artifacts",
                json=sample_artifact_request.model_dump(),
            )
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "art_123"
            assert data["status"] == "completed"

    def test_list_artifacts(self, test_client):
        """Test listing artifacts."""
        with patch("main.artifact_service") as mock_service:
            mock_service.list_artifacts = AsyncMock(return_value=[])

            response = test_client.get("/api/artifacts")
            assert response.status_code == 200
            assert isinstance(response.json(), list)

    def test_list_artifacts_by_type(self, test_client):
        """Test listing artifacts filtered by type."""
        with patch("main.artifact_service") as mock_service:
            from app.models.artifact import ArtifactType

            mock_service.list_artifacts = AsyncMock(return_value=[])

            response = test_client.get("/api/artifacts?artifact_type=pdf")
            assert response.status_code == 200

    def test_get_artifact(self, test_client):
        """Test getting artifact details."""
        with patch("main.artifact_service") as mock_service:
            from app.models.artifact import ArtifactResponse, ArtifactStatus, ArtifactType

            mock_response = ArtifactResponse(
                id="art_123",
                artifact_type=ArtifactType.PDF,
                status=ArtifactStatus.COMPLETED,
                filename="test.pdf",
                download_url="/download",
            )
            mock_service.get_artifact = AsyncMock(return_value=mock_response)

            response = test_client.get("/api/artifacts/art_123")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "art_123"

    def test_get_artifact_not_found(self, test_client):
        """Test getting non-existent artifact."""
        with patch("main.artifact_service") as mock_service:
            mock_service.get_artifact = AsyncMock(return_value=None)

            response = test_client.get("/api/artifacts/nonexistent")
            assert response.status_code == 404

    def test_download_artifact(self, test_client):
        """Test downloading an artifact."""
        with patch("main.artifact_service") as mock_service:
            mock_service.get_artifact_content = AsyncMock(return_value=b"test content")
            mock_service.get_artifact = AsyncMock(
                return_value=MagicMock(filename="test.txt")
            )

            response = test_client.get("/api/artifacts/art_123/download")
            assert response.status_code == 200
            assert response.content == b"test content"

    def test_delete_artifact(self, test_client):
        """Test deleting an artifact."""
        with patch("main.artifact_service") as mock_service:
            mock_service.delete_artifact = AsyncMock(return_value=True)

            response = test_client.delete("/api/artifacts/art_123")
            assert response.status_code == 204

    def test_delete_artifact_not_found(self, test_client):
        """Test deleting non-existent artifact."""
        with patch("main.artifact_service") as mock_service:
            mock_service.delete_artifact = AsyncMock(return_value=False)

            response = test_client.delete("/api/artifacts/nonexistent")
            assert response.status_code == 404


@pytest.mark.api
class TestComputerUseEndpoints:
    """Tests for computer use endpoints."""

    def test_execute_computer_use(self, test_client, sample_computer_use_request):
        """Test executing computer use actions."""
        with patch("main.computer_use_service") as mock_service:
            from app.models.computer_use import ActionResult

            mock_response = MagicMock()
            mock_response.session_id = "session_123"
            mock_response.results = [ActionResult(success=True, output="Done", execution_time_ms=100)]
            mock_response.screenshots = []
            mock_response.total_execution_time_ms = 100
            mock_response.actions_completed = 1
            mock_response.actions_failed = 0
            mock_response.metadata = {}
            mock_response.timestamp = MagicMock()

            mock_service.execute_actions = AsyncMock(return_value=mock_response)

            response = test_client.post(
                "/api/computer-use",
                json=sample_computer_use_request.model_dump(),
            )
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "session_123"

    def test_list_computer_sessions(self, test_client):
        """Test listing computer use sessions."""
        with patch("main.computer_use_service") as mock_service:
            mock_service.list_sessions = AsyncMock(return_value=[])

            response = test_client.get("/api/computer-use/sessions")
            assert response.status_code == 200
            data = response.json()
            assert "sessions" in data

    def test_close_computer_session(self, test_client):
        """Test closing a computer use session."""
        with patch("main.computer_use_service") as mock_service:
            mock_service.close_session = AsyncMock(return_value=True)

            response = test_client.delete("/api/computer-use/sessions/session_123")
            assert response.status_code == 200

    def test_close_nonexistent_session(self, test_client):
        """Test closing non-existent session."""
        with patch("main.computer_use_service") as mock_service:
            mock_service.close_session = AsyncMock(return_value=False)

            response = test_client.delete("/api/computer-use/sessions/nonexistent")
            assert response.status_code == 404

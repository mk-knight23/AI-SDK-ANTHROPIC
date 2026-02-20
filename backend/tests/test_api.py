"""
Tests for the FastAPI endpoints.
Tests all API endpoints including /synthesize and /query.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app, knowledge_index, document_ingestor


@pytest_asyncio.fixture
async def async_client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture(autouse=True)
async def clear_index():
    """Clear the knowledge index before each test."""
    knowledge_index.clear()
    yield
    knowledge_index.clear()


@pytest_asyncio.fixture
async def sample_documents(async_client):
    """Fixture to create sample documents for testing."""
    docs = [
        {
            "content": "Machine learning is a subset of artificial intelligence",
            "title": "ML Overview",
            "source": "arxiv",
            "metadata": {"author": "Researcher A"}
        },
        {
            "content": "Deep learning uses neural networks with multiple layers",
            "title": "Deep Learning Basics",
            "source": "arxiv",
            "metadata": {"author": "Researcher B"}
        },
        {
            "content": "Natural language processing enables machines to understand text",
            "title": "NLP Fundamentals",
            "source": "arxiv",
            "metadata": {"author": "Researcher C"}
        }
    ]

    for doc in docs:
        await async_client.post("/ingest", json=doc)

    return docs


class TestRootEndpoint:
    """Tests for the root endpoint."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """Test the root endpoint returns correct structure."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome to ResearchSynthesis API" in data["message"]
        assert "docs" in data
        assert "health" in data
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_root_endpoints_listed(self, async_client):
        """Test that root endpoint lists available endpoints."""
        response = await async_client.get("/")
        data = response.json()

        endpoints = data["endpoints"]
        assert "ingest" in endpoints
        assert "query" in endpoints
        assert "synthesize" in endpoints
        assert "documents" in endpoints


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test the health check endpoint."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"
        assert data["service"] == "research-synthesis-api"

    @pytest.mark.asyncio
    async def test_health_response_structure(self, async_client):
        """Test that health response has all required fields."""
        response = await async_client.get("/health")
        data = response.json()

        required_fields = ["status", "timestamp", "version", "service"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_health_timestamp_iso_format(self, async_client):
        """Test that timestamp is in ISO format."""
        response = await async_client.get("/health")
        data = response.json()

        # Should be able to parse the timestamp
        from datetime import datetime
        timestamp = data["timestamp"]
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Timestamp is not in valid ISO format: {timestamp}")


class TestIngestEndpoint:
    """Tests for the document ingestion endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_single_document(self, async_client):
        """Test ingesting a single document."""
        payload = {
            "content": "Test document content",
            "title": "Test Document",
            "source": "test",
            "metadata": {"key": "value"}
        }

        response = await async_client.post("/ingest", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["title"] == "Test Document"
        assert data["source"] == "test"
        assert "created_at" in data
        assert "message" in data

    @pytest.mark.asyncio
    async def test_ingest_without_metadata(self, async_client):
        """Test ingesting a document without optional metadata."""
        payload = {
            "content": "Simple content",
            "title": "Simple Title"
        }

        response = await async_client.post("/ingest", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Simple Title"

    @pytest.mark.asyncio
    async def test_ingest_empty_content(self, async_client):
        """Test that empty content is rejected."""
        payload = {
            "content": "",
            "title": "Test"
        }

        response = await async_client.post("/ingest", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_ingest_whitespace_content(self, async_client):
        """Test that whitespace-only content is rejected."""
        payload = {
            "content": "   ",
            "title": "Test"
        }

        response = await async_client.post("/ingest", json=payload)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_empty_title(self, async_client):
        """Test that empty title is rejected."""
        payload = {
            "content": "Content",
            "title": ""
        }

        response = await async_client.post("/ingest", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_ingest_document_added_to_index(self, async_client):
        """Test that ingested document is queryable."""
        payload = {
            "content": "Unique content for testing",
            "title": "Unique Title"
        }

        await async_client.post("/ingest", json=payload)

        # Query for the content
        query_response = await async_client.post("/query", json={"query": "unique content"})
        assert query_response.status_code == 200
        data = query_response.json()
        assert data["total_results"] >= 1


class TestBatchIngestEndpoint:
    """Tests for the batch ingestion endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_batch(self, async_client):
        """Test ingesting multiple documents at once."""
        payload = {
            "documents": [
                {"content": "Doc 1", "title": "Title 1", "source": "src1"},
                {"content": "Doc 2", "title": "Title 2", "source": "src2"},
                {"content": "Doc 3", "title": "Title 3", "source": "src3"}
            ]
        }

        response = await async_client.post("/ingest/batch", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["ingested_count"] == 3
        assert len(data["documents"]) == 3

    @pytest.mark.asyncio
    async def test_ingest_batch_empty(self, async_client):
        """Test that empty batch is rejected."""
        payload = {"documents": []}

        response = await async_client.post("/ingest/batch", json=payload)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_batch_partial_invalid(self, async_client):
        """Test that batch with invalid document is rejected."""
        payload = {
            "documents": [
                {"content": "Valid", "title": "Valid"},
                {"content": "", "title": "Invalid"}  # Empty content
            ]
        }

        response = await async_client.post("/ingest/batch", json=payload)

        assert response.status_code == 422  # Pydantic validation error


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    @pytest.mark.asyncio
    async def test_query_with_results(self, async_client, sample_documents):
        """Test querying returns relevant documents."""
        payload = {"query": "machine learning", "top_k": 5}

        response = await async_client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert data["total_results"] > 0
        assert len(data["results"]) > 0

    @pytest.mark.asyncio
    async def test_query_no_results(self, async_client, sample_documents):
        """Test querying with no matches."""
        payload = {"query": "quantum physics", "top_k": 5}

        response = await async_client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 0
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self, async_client, sample_documents):
        """Test that top_k parameter limits results."""
        payload = {"query": "research", "top_k": 2}

        response = await async_client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2

    @pytest.mark.asyncio
    async def test_query_empty_index(self, async_client):
        """Test querying with no documents."""
        payload = {"query": "anything"}

        response = await async_client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 0

    @pytest.mark.asyncio
    async def test_query_invalid_top_k(self, async_client):
        """Test that invalid top_k values are rejected."""
        payload = {"query": "test", "top_k": 0}

        response = await async_client.post("/query", json=payload)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_top_k_too_high(self, async_client):
        """Test that top_k above max is rejected."""
        payload = {"query": "test", "top_k": 25}

        response = await async_client.post("/query", json=payload)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_result_structure(self, async_client, sample_documents):
        """Test that query results have correct structure."""
        payload = {"query": "machine learning"}

        response = await async_client.post("/query", json=payload)
        data = response.json()

        if data["results"]:
            result = data["results"][0]
            required_fields = ["id", "title", "source", "content_preview", "metadata", "created_at"]
            for field in required_fields:
                assert field in result, f"Missing field: {field}"


class TestSynthesizeEndpoint:
    """Tests for the synthesize endpoint."""

    @pytest.mark.asyncio
    async def test_synthesize_with_results(self, async_client, sample_documents):
        """Test synthesis returns synthesized response."""
        payload = {"query": "artificial intelligence"}

        response = await async_client.post("/synthesize", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "artificial intelligence"
        assert "response" in data
        assert len(data["response"]) > 0
        assert "sources" in data
        assert "confidence" in data
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_synthesize_no_results(self, async_client, sample_documents):
        """Test synthesis with no matching documents."""
        payload = {"query": "quantum computing"}

        response = await async_client.post("/synthesize", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "quantum computing"
        assert data["sources"] == []
        assert data["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_source_structure(self, async_client, sample_documents):
        """Test that synthesis sources have correct structure."""
        payload = {"query": "machine learning"}

        response = await async_client.post("/synthesize", json=payload)
        data = response.json()

        for source in data["sources"]:
            assert "id" in source
            assert "title" in source
            assert "source" in source

    @pytest.mark.asyncio
    async def test_synthesize_confidence_range(self, async_client, sample_documents):
        """Test that confidence is within valid range."""
        payload = {"query": "research"}

        response = await async_client.post("/synthesize", json=payload)
        data = response.json()

        assert 0 <= data["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_synthesize_empty_query(self, async_client):
        """Test that empty query is rejected."""
        payload = {"query": ""}

        response = await async_client.post("/synthesize", json=payload)

        assert response.status_code == 422


class TestDocumentsListEndpoint:
    """Tests for the documents list endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, async_client):
        """Test listing documents when empty."""
        response = await async_client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_list_documents_with_data(self, async_client, sample_documents):
        """Test listing documents with data."""
        response = await async_client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    @pytest.mark.asyncio
    async def test_list_documents_structure(self, async_client, sample_documents):
        """Test that listed documents have correct structure."""
        response = await async_client.get("/documents")
        data = response.json()

        for doc in data:
            required_fields = ["id", "title", "source", "content_preview", "metadata", "created_at"]
            for field in required_fields:
                assert field in doc, f"Missing field: {field}"


class TestDocumentGetEndpoint:
    """Tests for getting a specific document."""

    @pytest.mark.asyncio
    async def test_get_document_exists(self, async_client, sample_documents):
        """Test getting an existing document."""
        # First list to get an ID
        list_response = await async_client.get("/documents")
        docs = list_response.json()
        doc_id = docs[0]["id"]

        response = await async_client.get(f"/documents/{doc_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id

    @pytest.mark.asyncio
    async def test_get_document_not_exists(self, async_client):
        """Test getting a non-existent document."""
        response = await async_client.get("/documents/nonexistent-id")

        assert response.status_code == 404


class TestDocumentDeleteEndpoint:
    """Tests for deleting a document."""

    @pytest.mark.asyncio
    async def test_delete_document_exists(self, async_client, sample_documents):
        """Test deleting an existing document."""
        # First list to get an ID
        list_response = await async_client.get("/documents")
        docs = list_response.json()
        doc_id = docs[0]["id"]

        response = await async_client.delete(f"/documents/{doc_id}")

        assert response.status_code == 204

        # Verify it's gone
        get_response = await async_client.get(f"/documents/{doc_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_not_exists(self, async_client):
        """Test deleting a non-existent document."""
        response = await async_client.delete("/documents/nonexistent-id")

        assert response.status_code == 404


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_404_handler(self, async_client):
        """Test 404 for non-existent endpoint."""
        response = await async_client.get("/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_json(self, async_client):
        """Test handling of invalid JSON."""
        response = await async_client.post(
            "/ingest",
            content="not valid json",
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 422

"""
Tests for the Knowledge Graph module.
Tests document ingestion, indexing, and query functionality.
"""

import pytest
from datetime import datetime
from knowledge_graph import (
    Document,
    KnowledgeGraphIndex,
    QueryEngine,
    DocumentIngestor,
    SynthesisResult,
)


class TestDocument:
    """Tests for the Document dataclass."""

    def test_document_creation(self):
        """Test creating a document with all fields."""
        doc = Document(
            id="test-id",
            content="Test content",
            title="Test Title",
            source="test_source",
            metadata={"author": "Test Author"}
        )

        assert doc.id == "test-id"
        assert doc.content == "Test content"
        assert doc.title == "Test Title"
        assert doc.source == "test_source"
        assert doc.metadata == {"author": "Test Author"}
        assert isinstance(doc.created_at, datetime)

    def test_document_auto_id_generation(self):
        """Test that document ID is auto-generated from content hash."""
        content = "Unique content for hashing"
        doc = Document(
            id="",
            content=content,
            title="Test",
            source="test"
        )

        assert doc.id != ""
        assert len(doc.id) == 32  # MD5 hash length

    def test_document_same_content_same_id(self):
        """Test that same content produces same ID."""
        content = "Same content"
        doc1 = Document(id="", content=content, title="Title", source="source")
        doc2 = Document(id="", content=content, title="Different Title", source="other")

        assert doc1.id == doc2.id

    def test_document_different_content_different_id(self):
        """Test that different content produces different IDs."""
        doc1 = Document(id="", content="Content A", title="Title", source="source")
        doc2 = Document(id="", content="Content B", title="Title", source="source")

        assert doc1.id != doc2.id


class TestKnowledgeGraphIndex:
    """Tests for the KnowledgeGraphIndex class."""

    @pytest.fixture
    def empty_index(self):
        """Fixture for an empty knowledge graph index."""
        return KnowledgeGraphIndex()

    @pytest.fixture
    def populated_index(self):
        """Fixture for a populated knowledge graph index."""
        index = KnowledgeGraphIndex()
        docs = [
            Document(id="doc1", content="Machine learning is fascinating", title="ML Intro", source="arxiv"),
            Document(id="doc2", content="Deep learning revolutionizes AI", title="Deep Learning", source="arxiv"),
            Document(id="doc3", content="Neural networks are powerful", title="Neural Nets", source="arxiv"),
        ]
        index.add_documents(docs)
        return index

    def test_empty_index_initialization(self, empty_index):
        """Test that a new index is empty and not indexed."""
        assert empty_index.document_count() == 0
        assert empty_index.indexed is False
        assert empty_index.list_documents() == []

    def test_add_single_document(self, empty_index):
        """Test adding a single document to the index."""
        doc = Document(id="doc1", content="Test content", title="Test", source="test")
        empty_index.add_documents([doc])

        assert empty_index.document_count() == 1
        assert empty_index.indexed is True
        assert empty_index.get_document("doc1") == doc

    def test_add_multiple_documents(self, empty_index):
        """Test adding multiple documents at once."""
        docs = [
            Document(id="doc1", content="Content 1", title="Title 1", source="source"),
            Document(id="doc2", content="Content 2", title="Title 2", source="source"),
        ]
        empty_index.add_documents(docs)

        assert empty_index.document_count() == 2
        assert empty_index.get_document("doc1") is not None
        assert empty_index.get_document("doc2") is not None

    def test_query_with_no_documents(self, empty_index):
        """Test querying an empty index returns empty results."""
        results = empty_index.query("machine learning")
        assert results == []

    def test_query_finds_relevant_documents(self, populated_index):
        """Test that query finds documents with matching keywords."""
        results = populated_index.query("machine learning")

        assert len(results) >= 1
        assert any("machine" in doc.content.lower() for doc in results)

    def test_query_returns_top_k_results(self, populated_index):
        """Test that query respects top_k parameter."""
        results = populated_index.query("learning", top_k=2)

        assert len(results) <= 2

    def test_query_ranking(self, populated_index):
        """Test that documents are ranked by relevance."""
        # Add a document with multiple matching terms
        doc = Document(
            id="doc4",
            content="Machine learning and deep learning are both important",
            title="Combined",
            source="arxiv"
        )
        populated_index.add_documents([doc])

        results = populated_index.query("machine learning deep")

        # The document with most matches should be first
        assert results[0].id == "doc4"

    def test_get_document_exists(self, populated_index):
        """Test retrieving an existing document."""
        doc = populated_index.get_document("doc1")

        assert doc is not None
        assert doc.id == "doc1"
        assert doc.title == "ML Intro"

    def test_get_document_not_exists(self, populated_index):
        """Test retrieving a non-existent document returns None."""
        doc = populated_index.get_document("nonexistent")

        assert doc is None

    def test_list_documents(self, populated_index):
        """Test listing all documents."""
        docs = populated_index.list_documents()

        assert len(docs) == 3
        doc_ids = {doc.id for doc in docs}
        assert doc_ids == {"doc1", "doc2", "doc3"}

    def test_clear_index(self, populated_index):
        """Test clearing all documents from the index."""
        populated_index.clear()

        assert populated_index.document_count() == 0
        assert populated_index.indexed is False
        assert populated_index.list_documents() == []

    def test_document_count(self, populated_index):
        """Test document count is accurate."""
        assert populated_index.document_count() == 3

        populated_index.add_documents([
            Document(id="doc4", content="New", title="New", source="source")
        ])

        assert populated_index.document_count() == 4


class TestQueryEngine:
    """Tests for the QueryEngine class."""

    @pytest.fixture
    def query_engine_with_docs(self):
        """Fixture for query engine with sample documents."""
        index = KnowledgeGraphIndex()
        docs = [
            Document(id="doc1", content="Machine learning basics", title="ML Basics", source="arxiv"),
            Document(id="doc2", content="Deep learning advances", title="DL Advances", source="arxiv"),
        ]
        index.add_documents(docs)
        return QueryEngine(index)

    def test_synthesize_with_results(self, query_engine_with_docs):
        """Test synthesis returns result with sources."""
        result = query_engine_with_docs.synthesize("machine learning")

        assert isinstance(result, SynthesisResult)
        assert result.query == "machine learning"
        assert len(result.response) > 0
        assert len(result.sources) > 0
        assert result.confidence > 0
        assert result.metadata["status"] == "success"

    def test_synthesize_no_results(self, query_engine_with_docs):
        """Test synthesis with no matching documents."""
        result = query_engine_with_docs.synthesize("quantum physics")

        assert isinstance(result, SynthesisResult)
        assert result.query == "quantum physics"
        assert "no relevant" in result.response.lower() or "No relevant" in result.response
        assert result.sources == []
        assert result.confidence == 0.0
        assert result.metadata["status"] == "no_results"

    def test_synthesize_with_provided_context(self, query_engine_with_docs):
        """Test synthesis with explicitly provided context documents."""
        context_docs = [
            Document(id="ctx1", content="Custom context", title="Context", source="manual")
        ]
        result = query_engine_with_docs.synthesize("test query", context_docs=context_docs)

        assert len(result.sources) == 1
        assert result.sources[0].id == "ctx1"

    def test_synthesize_confidence_calculation(self, query_engine_with_docs):
        """Test that confidence is calculated based on number of sources."""
        result = query_engine_with_docs.synthesize("machine learning")

        # Confidence should be between 0.5 and 0.95
        assert 0.5 <= result.confidence <= 0.95

    def test_generate_response_with_content(self, query_engine_with_docs):
        """Test response generation with content pieces."""
        content = ["Finding 1: ML is great", "Finding 2: DL is powerful"]
        response = query_engine_with_docs._generate_response("test", content)

        assert "test" in response
        assert "Finding 1" in response
        assert "Finding 2" in response
        assert "Key findings:" in response

    def test_generate_response_empty(self, query_engine_with_docs):
        """Test response generation with no content."""
        response = query_engine_with_docs._generate_response("test", [])

        assert "No relevant information" in response


class TestDocumentIngestor:
    """Tests for the DocumentIngestor class."""

    @pytest.fixture
    def ingestor(self):
        """Fixture for document ingestor."""
        index = KnowledgeGraphIndex()
        return DocumentIngestor(index)

    def test_ingest_text(self, ingestor):
        """Test ingesting a text document."""
        doc = ingestor.ingest_text(
            content="Test content",
            title="Test Title",
            source="test_source",
            metadata={"key": "value"}
        )

        assert doc.content == "Test content"
        assert doc.title == "Test Title"
        assert doc.source == "test_source"
        assert doc.metadata == {"key": "value"}
        assert doc.id != ""

    def test_ingest_text_adds_to_index(self, ingestor):
        """Test that ingested text is added to the index."""
        doc = ingestor.ingest_text("Content", "Title", "source")

        assert ingestor.index.document_count() == 1
        assert ingestor.index.get_document(doc.id) == doc

    def test_ingest_batch(self, ingestor):
        """Test batch ingestion of documents."""
        docs_data = [
            {"content": "Content 1", "title": "Title 1", "source": "src1"},
            {"content": "Content 2", "title": "Title 2", "source": "src2", "metadata": {"a": 1}},
        ]
        documents = ingestor.ingest_batch(docs_data)

        assert len(documents) == 2
        assert ingestor.index.document_count() == 2
        assert documents[0].title == "Title 1"
        assert documents[1].metadata == {"a": 1}

    def test_ingest_batch_empty(self, ingestor):
        """Test batch ingestion with empty list."""
        documents = ingestor.ingest_batch([])

        assert documents == []
        assert ingestor.index.document_count() == 0

    def test_validate_document_valid(self, ingestor):
        """Test validation of valid document."""
        is_valid, error = ingestor.validate_document("Valid content", "Valid title")

        assert is_valid is True
        assert error is None

    def test_validate_document_empty_content(self, ingestor):
        """Test validation rejects empty content."""
        is_valid, error = ingestor.validate_document("", "Title")

        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_document_whitespace_content(self, ingestor):
        """Test validation rejects whitespace-only content."""
        is_valid, error = ingestor.validate_document("   ", "Title")

        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_document_empty_title(self, ingestor):
        """Test validation rejects empty title."""
        is_valid, error = ingestor.validate_document("Content", "")

        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_document_too_large(self, ingestor):
        """Test validation rejects oversized content."""
        large_content = "x" * 10_000_001  # Just over 10MB
        is_valid, error = ingestor.validate_document(large_content, "Title")

        assert is_valid is False
        assert "exceeds" in error.lower()


class TestIntegration:
    """Integration tests for the full knowledge graph workflow."""

    def test_full_workflow(self):
        """Test complete workflow: ingest -> query -> synthesize."""
        # Setup
        index = KnowledgeGraphIndex()
        ingestor = DocumentIngestor(index)
        query_engine = QueryEngine(index)

        # Ingest documents
        docs = [
            {"content": f"Research paper about topic {i}", "title": f"Paper {i}", "source": "arxiv"}
            for i in range(5)
        ]
        ingestor.ingest_batch(docs)

        # Query
        results = index.query("research paper", top_k=3)
        assert len(results) > 0

        # Synthesize
        synthesis = query_engine.synthesize("research paper")
        assert synthesis.confidence > 0
        assert len(synthesis.sources) > 0

    def test_multiple_ingestions_accumulate(self):
        """Test that multiple ingestion calls accumulate documents."""
        index = KnowledgeGraphIndex()
        ingestor = DocumentIngestor(index)

        ingestor.ingest_text("Content 1", "Title 1", "source")
        assert index.document_count() == 1

        ingestor.ingest_text("Content 2", "Title 2", "source")
        assert index.document_count() == 2

        ingestor.ingest_batch([
            {"content": "C3", "title": "T3", "source": "s"},
            {"content": "C4", "title": "T4", "source": "s"},
        ])
        assert index.document_count() == 4

"""
Knowledge Graph module using LlamaIndex for research synthesis.
Provides document ingestion, indexing, and query capabilities.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import os


@dataclass
class Document:
    """Represents a research document."""
    id: str
    content: str
    title: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class SynthesisResult:
    """Result of a synthesis operation."""
    query: str
    response: str
    sources: List[Document]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphIndex:
    """
    In-memory knowledge graph index for document storage and retrieval.
    This is a simplified implementation for testing purposes.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.indexed: bool = False

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        for doc in documents:
            self.documents[doc.id] = doc
        self.indexed = True

    def query(self, query_text: str, top_k: int = 5) -> List[Document]:
        """
        Simple keyword-based query for demonstration.
        In production, this would use vector similarity search.
        """
        if not self.indexed:
            return []

        query_terms = query_text.lower().split()
        scored_docs = []

        for doc in self.documents.values():
            score = 0
            content_lower = doc.content.lower()
            for term in query_terms:
                if term in content_lower:
                    score += 1
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def list_documents(self) -> List[Document]:
        """List all documents in the index."""
        return list(self.documents.values())

    def clear(self) -> None:
        """Clear all documents from the index."""
        self.documents.clear()
        self.indexed = False

    def document_count(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)


class QueryEngine:
    """
    Query engine for synthesizing research from documents.
    """

    def __init__(self, index: KnowledgeGraphIndex):
        self.index = index

    def synthesize(self, query: str, context_docs: Optional[List[Document]] = None) -> SynthesisResult:
        """
        Synthesize a response from documents based on the query.
        """
        if context_docs is None:
            context_docs = self.index.query(query)

        if not context_docs:
            return SynthesisResult(
                query=query,
                response="No relevant documents found for this query.",
                sources=[],
                confidence=0.0,
                metadata={"status": "no_results"}
            )

        # Simple synthesis - in production this would use LLM
        relevant_content = []
        for doc in context_docs:
            relevant_content.append(f"From '{doc.title}': {doc.content[:200]}...")

        synthesized_response = self._generate_response(query, relevant_content)
        confidence = min(0.5 + (len(context_docs) * 0.1), 0.95)

        return SynthesisResult(
            query=query,
            response=synthesized_response,
            sources=context_docs,
            confidence=confidence,
            metadata={
                "status": "success",
                "document_count": len(context_docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def _generate_response(self, query: str, content_pieces: List[str]) -> str:
        """Generate a synthesized response from content pieces."""
        if not content_pieces:
            return "No relevant information found."

        response_parts = [
            f"Based on the research documents, here's the synthesis for: '{query}'",
            "",
            "Key findings:"
        ]

        for i, content in enumerate(content_pieces, 1):
            response_parts.append(f"{i}. {content}")

        response_parts.extend([
            "",
            "This synthesis combines insights from multiple sources."
        ])

        return "\n".join(response_parts)


class DocumentIngestor:
    """
    Handles document ingestion into the knowledge graph.
    """

    def __init__(self, index: KnowledgeGraphIndex):
        self.index = index

    def ingest_text(self, content: str, title: str, source: str = "manual",
                    metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Ingest a text document into the knowledge graph."""
        doc = Document(
            id="",
            content=content,
            title=title,
            source=source,
            metadata=metadata or {}
        )
        self.index.add_documents([doc])
        return doc

    def ingest_batch(self, documents_data: List[Dict[str, Any]]) -> List[Document]:
        """Ingest multiple documents at once."""
        documents = []
        for data in documents_data:
            doc = Document(
                id="",
                content=data["content"],
                title=data["title"],
                source=data.get("source", "batch"),
                metadata=data.get("metadata", {})
            )
            documents.append(doc)

        self.index.add_documents(documents)
        return documents

    def validate_document(self, content: str, title: str) -> tuple[bool, Optional[str]]:
        """Validate a document before ingestion."""
        if not content or not content.strip():
            return False, "Document content cannot be empty"

        if not title or not title.strip():
            return False, "Document title cannot be empty"

        if len(content) > 10_000_000:  # 10MB limit
            return False, "Document content exceeds maximum size (10MB)"

        return True, None

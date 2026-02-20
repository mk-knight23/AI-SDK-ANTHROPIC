"""
ResearchSynthesis Backend API
FastAPI application with knowledge graph, document ingestion, and synthesis endpoints.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import os

from knowledge_graph import (
    KnowledgeGraphIndex,
    QueryEngine,
    DocumentIngestor,
    Document,
    SynthesisResult,
)

app = FastAPI(
    title="ResearchSynthesis API",
    description="AI-powered research synthesis platform backend with LlamaIndex knowledge graph",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global knowledge graph components
knowledge_index = KnowledgeGraphIndex()
query_engine = QueryEngine(knowledge_index)
document_ingestor = DocumentIngestor(knowledge_index)


# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    service: str


class DocumentIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Document content")
    title: str = Field(..., min_length=1, description="Document title")
    source: str = Field(default="api", description="Document source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class DocumentIngestResponse(BaseModel):
    id: str
    title: str
    source: str
    created_at: str
    message: str


class DocumentResponse(BaseModel):
    id: str
    title: str
    source: str
    content_preview: str
    metadata: Dict[str, Any]
    created_at: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class QueryResponse(BaseModel):
    query: str
    results: List[DocumentResponse]
    total_results: int


class SynthesizeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Synthesis query")


class SourceReference(BaseModel):
    id: str
    title: str
    source: str


class SynthesizeResponse(BaseModel):
    query: str
    response: str
    sources: List[SourceReference]
    confidence: float
    metadata: Dict[str, Any]


class BatchIngestRequest(BaseModel):
    documents: List[DocumentIngestRequest]


class BatchIngestResponse(BaseModel):
    ingested_count: int
    documents: List[DocumentIngestResponse]


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to ResearchSynthesis API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest": "POST /ingest",
            "query": "POST /query",
            "synthesize": "POST /synthesize",
            "documents": "GET /documents",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="0.1.0",
        service="research-synthesis-api",
    )


@app.post("/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(request: DocumentIngestRequest):
    """
    Ingest a document into the knowledge graph.
    """
    # Validate document
    is_valid, error_message = document_ingestor.validate_document(
        request.content, request.title
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )

    # Ingest document
    doc = document_ingestor.ingest_text(
        content=request.content,
        title=request.title,
        source=request.source,
        metadata=request.metadata
    )

    return DocumentIngestResponse(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        created_at=doc.created_at.isoformat(),
        message="Document ingested successfully"
    )


@app.post("/ingest/batch", response_model=BatchIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_batch(request: BatchIngestRequest):
    """
    Ingest multiple documents in a batch.
    """
    if not request.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents provided"
        )

    # Validate all documents first
    for i, doc_request in enumerate(request.documents):
        is_valid, error_message = document_ingestor.validate_document(
            doc_request.content, doc_request.title
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document {i + 1}: {error_message}"
            )

    # Convert to dict format for ingestion
    docs_data = [
        {
            "content": d.content,
            "title": d.title,
            "source": d.source,
            "metadata": d.metadata
        }
        for d in request.documents
    ]

    documents = document_ingestor.ingest_batch(docs_data)

    return BatchIngestResponse(
        ingested_count=len(documents),
        documents=[
            DocumentIngestResponse(
                id=doc.id,
                title=doc.title,
                source=doc.source,
                created_at=doc.created_at.isoformat(),
                message="Document ingested successfully"
            )
            for doc in documents
        ]
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge graph for relevant documents.
    """
    results = knowledge_index.query(request.query, top_k=request.top_k)

    return QueryResponse(
        query=request.query,
        results=[
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                source=doc.source,
                content_preview=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                metadata=doc.metadata,
                created_at=doc.created_at.isoformat()
            )
            for doc in results
        ],
        total_results=len(results)
    )


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize a response from the knowledge graph based on the query.
    """
    result = query_engine.synthesize(request.query)

    return SynthesizeResponse(
        query=result.query,
        response=result.response,
        sources=[
            SourceReference(id=doc.id, title=doc.title, source=doc.source)
            for doc in result.sources
        ],
        confidence=result.confidence,
        metadata=result.metadata
    )


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    """
    List all documents in the knowledge graph.
    """
    documents = knowledge_index.list_documents()

    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            source=doc.source,
            content_preview=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            metadata=doc.metadata,
            created_at=doc.created_at.isoformat()
        )
        for doc in documents
    ]


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """
    Get a specific document by ID.
    """
    doc = knowledge_index.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found"
        )

    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        source=doc.source,
        content_preview=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
        metadata=doc.metadata,
        created_at=doc.created_at.isoformat()
    )


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str):
    """
    Delete a document from the knowledge graph.
    Note: This is a simplified implementation. In production,
    use proper deletion with confirmation.
    """
    doc = knowledge_index.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found"
        )

    # Remove from index
    del knowledge_index.documents[doc_id]
    return None


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

"""
Anthropic AI SDK - FastAPI Backend

Main application with endpoints for:
- Chat completion (with extended thinking and streaming)
- Artifact generation (PDF, code, etc.)
- Computer use capabilities
- Provider management and health checks
"""

import os
import asyncio
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import ValidationError

from app.models import (
    HealthResponse,
    ChatRequest,
    ChatResponse,
    StreamingChunk,
    ArtifactCreateRequest,
    ArtifactResponse,
    ArtifactType,
    ComputerUseRequest,
    ComputerUseResponse,
    ProviderStatus,
    ProviderSwitchRequest,
)

from app.services import (
    LLMRouter,
    ArtifactService,
    ComputerUseService,
)

# Initialize services
llm_router = LLMRouter()
artifact_service = ArtifactService()
computer_use_service = ComputerUseService()

# Create FastAPI app
app = FastAPI(
    title="Anthropic AI SDK API",
    description="FastAPI backend for Anthropic Claude with extended thinking, streaming, computer use, and artifact generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Anthropic AI SDK API",
        "version": "1.0.0",
        "description": "FastAPI backend for Anthropic Claude integration",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /api/chat",
            "stream": "POST /api/chat/stream",
            "artifacts": "POST /api/artifacts",
            "computer_use": "POST /api/computer-use",
            "providers": "GET /api/providers",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    provider_status = await llm_router.health_check()

    # Determine overall health
    is_healthy = any(provider_status.values())

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        service="anthropic-api",
        environment=os.getenv("ENVIRONMENT", "development"),
        provider="anthropic" if provider_status.get("anthropic") else "openrouter",
    )


# ============================================================================
# Provider Management Endpoints
# ============================================================================

@app.get("/api/providers", response_model=List[ProviderStatus])
async def get_providers():
    """Get status of all LLM providers."""
    return await llm_router.get_provider_status()


@app.post("/api/providers/switch")
async def switch_provider(request: ProviderSwitchRequest):
    """Switch to a specific provider."""
    try:
        if request.force:
            llm_router.force_provider_switch(request.provider)
        else:
            llm_router.set_preferred_provider(request.provider)

        return {
            "message": f"Switched to {request.provider}",
            "provider": request.provider,
            "forced": request.force,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/providers/health")
async def check_providers_health():
    """Run health checks on all providers."""
    results = await llm_router.health_check()
    return {
        "providers": results,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Send a chat completion request.

    Supports:
    - Standard message completion
    - Extended thinking (chain-of-thought)
    - Tool/function calling
    - Automatic provider failover
    """
    try:
        response = await llm_router.chat(request)
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}",
        )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a streaming chat completion request.

    Returns Server-Sent Events (SSE) with streaming chunks.
    """
    if not request.stream:
        request.stream = True  # Force streaming

    async def generate():
        try:
            async for chunk in llm_router.chat_stream(request):
                # Format as SSE
                data = chunk.model_dump_json(exclude_none=True)
                yield f"data: {data}\n\n"

            # Send completion event
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {
                "event_type": "error",
                "error": str(e),
            }
            import json
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Artifact Endpoints
# ============================================================================

@app.post("/api/artifacts", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
async def create_artifact(request: ArtifactCreateRequest):
    """
    Create an artifact from content.

    Supports: PDF, code, markdown, HTML, text
    """
    try:
        artifact = await artifact_service.create_artifact(request)
        return artifact
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Artifact creation failed: {str(e)}",
        )


@app.get("/api/artifacts", response_model=List[ArtifactResponse])
async def list_artifacts(artifact_type: Optional[ArtifactType] = None):
    """List all artifacts, optionally filtered by type."""
    try:
        artifacts = await artifact_service.list_artifacts(artifact_type)
        return artifacts
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list artifacts: {str(e)}",
        )


@app.get("/api/artifacts/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(artifact_id: str):
    """Get artifact details by ID."""
    artifact = await artifact_service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found",
        )
    return artifact


@app.get("/api/artifacts/{artifact_id}/download")
async def download_artifact(artifact_id: str):
    """Download an artifact file."""
    content = await artifact_service.get_artifact_content(artifact_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found",
        )

    # Get artifact details for filename
    artifact = await artifact_service.get_artifact(artifact_id)
    if artifact:
        filename = artifact.filename
    else:
        filename = f"artifact_{artifact_id}"

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


@app.delete("/api/artifacts/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact(artifact_id: str):
    """Delete an artifact."""
    success = await artifact_service.delete_artifact(artifact_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found",
        )
    return None


# ============================================================================
# Computer Use Endpoints
# ============================================================================

@app.post("/api/computer-use", response_model=ComputerUseResponse)
async def execute_computer_use(request: ComputerUseRequest):
    """
    Execute computer use actions.

    Supports: click, type, scroll, screenshot, navigate, wait, drag
    """
    try:
        response = await computer_use_service.execute_actions(request)
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Computer use execution failed: {str(e)}",
        )


@app.get("/api/computer-use/sessions")
async def list_computer_sessions():
    """List active computer use sessions."""
    sessions = await computer_use_service.list_sessions()
    return {
        "sessions": sessions,
        "count": len(sessions),
    }


@app.delete("/api/computer-use/sessions/{session_id}")
async def close_computer_session(session_id: str):
    """Close a computer use session."""
    success = await computer_use_service.close_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return {"message": f"Session {session_id} closed"}


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Run initial health checks
    await llm_router.health_check()

    # Ensure artifact storage directory exists
    storage_dir = os.getenv("ARTIFACT_STORAGE_DIR", "./artifacts")
    os.makedirs(storage_dir, exist_ok=True)

    print("Anthropic AI SDK API started successfully")
    print(f"Documentation available at: /docs")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )

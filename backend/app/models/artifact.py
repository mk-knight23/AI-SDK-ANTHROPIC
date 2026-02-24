"""Artifact generation models."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ArtifactType(str, Enum):
    """Artifact types."""

    PDF = "pdf"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class ArtifactStatus(str, Enum):
    """Artifact generation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactCreateRequest(BaseModel):
    """Request to create an artifact."""

    content: str = Field(..., description="Content to generate artifact from")
    artifact_type: ArtifactType = Field(..., description="Type of artifact to generate")
    filename: Optional[str] = Field(None, description="Optional filename")
    title: Optional[str] = Field(None, description="Optional artifact title")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Artifact-specific options"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "# Hello World\n\nThis is a sample document.",
                "artifact_type": "pdf",
                "title": "Sample PDF",
                "filename": "sample.pdf",
            }
        }


class ArtifactResponse(BaseModel):
    """Artifact generation response."""

    id: str = Field(..., description="Unique artifact ID")
    artifact_type: ArtifactType = Field(..., description="Artifact type")
    status: ArtifactStatus = Field(..., description="Generation status")
    title: Optional[str] = Field(None, description="Artifact title")
    filename: str = Field(..., description="Generated filename")
    download_url: Optional[str] = Field(None, description="Download URL if completed")
    content_preview: Optional[str] = Field(None, description="Content preview")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Artifact metadata")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "art_123456",
                "artifact_type": "pdf",
                "status": "completed",
                "title": "Sample PDF",
                "filename": "sample_20250223.pdf",
                "download_url": "/api/artifacts/art_123456/download",
                "file_size_bytes": 12345,
            }
        }


class CodeArtifactOptions(BaseModel):
    """Options specific to code artifacts."""

    language: str = Field(default="python", description="Programming language")
    include_line_numbers: bool = Field(default=False, description="Add line numbers")
    include_syntax_highlighting: bool = Field(
        default=True, description="Enable syntax highlighting"
    )


class PDFArtifactOptions(BaseModel):
    """Options specific to PDF artifacts."""

    page_size: Literal["A4", "Letter", "Legal"] = Field(default="A4", description="Page size")
    margin: str = Field(default="1in", description="Page margins")
    font: str = Field(default="Helvetica", description="Font family")
    font_size: int = Field(default=12, ge=8, le=72, description="Font size")
    include_page_numbers: bool = Field(default=True, description="Add page numbers")
    title: Optional[str] = Field(None, description="PDF title metadata")
    author: Optional[str] = Field(None, description="PDF author metadata")

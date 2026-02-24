"""Health check models."""

from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current ISO timestamp")
    version: str = Field(..., description="API version")
    service: str = Field(..., description="Service name")
    environment: str = Field(default="development", description="Deployment environment")
    provider: str = Field(default="anthropic", description="Current LLM provider")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-02-23T10:30:00Z",
                "version": "1.0.0",
                "service": "anthropic-api",
                "environment": "production",
                "provider": "anthropic",
            }
        }

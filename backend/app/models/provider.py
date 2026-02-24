"""Provider management models."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime


class ProviderStatus(BaseModel):
    """Status of an LLM provider."""

    name: str = Field(..., description="Provider name")
    available: bool = Field(..., description="Whether provider is available")
    latency_ms: Optional[int] = Field(None, description="Average latency")
    error_message: Optional[str] = Field(None, description="Error if unavailable")
    models: list[str] = Field(default_factory=list, description="Available models")
    last_check: datetime = Field(
        default_factory=datetime.utcnow, description="Last health check time"
    )


class ProviderSwitchRequest(BaseModel):
    """Request to switch providers."""

    provider: Literal["anthropic", "openrouter"] = Field(
        ..., description="Provider to switch to"
    )
    force: bool = Field(default=False, description="Force switch even if provider is unhealthy")
    reason: Optional[str] = Field(None, description="Reason for switch")

"""Computer use capability models."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Computer action types."""

    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY = "key"
    DRAG = "drag"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    NAVIGATE = "navigate"


class ComputerAction(BaseModel):
    """A computer use action."""

    type: ActionType = Field(..., description="Action type")
    coordinates: Optional[Dict[str, int]] = Field(
        None, description="Screen coordinates {x, y}"
    )
    button: Optional[Literal["left", "middle", "right"]] = Field(
        None, description="Mouse button for click"
    )
    text: Optional[str] = Field(None, description="Text to type")
    key: Optional[str] = Field(None, description="Key combination")
    direction: Optional[Literal["up", "down", "left", "right"]] = Field(
        None, description="Scroll/drag direction"
    )
    amount: Optional[int] = Field(None, ge=1, description="Scroll/drag amount")
    duration_ms: Optional[int] = Field(None, ge=100, description="Wait duration")
    url: Optional[str] = Field(None, description="URL to navigate to")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional action metadata"
    )


class ActionResult(BaseModel):
    """Result of a computer action."""

    success: bool = Field(..., description="Whether action succeeded")
    output: Optional[str] = Field(None, description="Action output/result")
    error: Optional[str] = Field(None, description="Error message if failed")
    screenshot_data: Optional[str] = Field(None, description="Base64 screenshot if taken")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")


class ComputerUseRequest(BaseModel):
    """Request to perform computer use operations."""

    actions: List[ComputerAction] = Field(
        ..., min_items=1, description="Actions to perform"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for stateful operations"
    )
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="Timeout per action")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    viewport: Optional[Dict[str, int]] = Field(
        None, description="Browser viewport {width, height}"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "actions": [
                    {
                        "type": "navigate",
                        "url": "https://example.com",
                    },
                    {
                        "type": "screenshot",
                    },
                ],
                "headless": True,
                "viewport": {"width": 1920, "height": 1080},
            }
        }


class ComputerUseResponse(BaseModel):
    """Response from computer use operations."""

    session_id: str = Field(..., description="Session identifier")
    results: List[ActionResult] = Field(..., description="Results for each action")
    screenshots: List[str] = Field(default_factory=list, description="Base64 screenshots")
    total_execution_time_ms: int = Field(..., description="Total execution time")
    actions_completed: int = Field(..., description="Number of successful actions")
    actions_failed: int = Field(..., description="Number of failed actions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

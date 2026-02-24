"""
LLM Router service for managing primary and fallback providers.

Handles provider selection, automatic failover, and health monitoring.
"""

import asyncio
from typing import AsyncIterator, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    StreamingChunk,
)
from app.models.provider import ProviderStatus
from app.services.anthropic_service import AnthropicService
from app.services.openrouter_service import OpenRouterService


class ProviderHealth:
    """Health tracking for a provider."""

    def __init__(self, name: str):
        self.name = name
        self.available = True
        self.last_check = datetime.utcnow()
        self.failure_count = 0
        self.last_error: Optional[str] = None
        self.latency_samples: List[float] = []
        self.max_samples = 10

    def record_success(self, latency_ms: int):
        """Record a successful request."""
        self.available = True
        self.last_check = datetime.utcnow()
        self.failure_count = 0
        self.last_error = None
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_samples:
            self.latency_samples.pop(0)

    def record_failure(self, error: str):
        """Record a failed request."""
        self.failure_count += 1
        self.last_check = datetime.utcnow()
        self.last_error = error

        # Mark unavailable after 3 consecutive failures
        if self.failure_count >= 3:
            self.available = False

    def get_average_latency(self) -> Optional[int]:
        """Get average latency in milliseconds."""
        if not self.latency_samples:
            return None
        return int(sum(self.latency_samples) / len(self.latency_samples))

    def to_status(self, models: List[str]) -> ProviderStatus:
        """Convert to ProviderStatus model."""
        return ProviderStatus(
            name=self.name,
            available=self.available,
            latency_ms=self.get_average_latency(),
            error_message=self.last_error if not self.available else None,
            models=models,
            last_check=self.last_check,
        )


class LLMRouter:
    """
    Router for managing multiple LLM providers with automatic failover.

    Features:
    - Primary provider (Anthropic) with automatic fallback
    - Health monitoring and tracking
    - Automatic recovery of failed providers
    - Latency tracking
    """

    HEALTH_CHECK_INTERVAL = 60  # seconds
    MAX_FAILURES_BEFORE_SWITCH = 3

    def __init__(
        self,
        anthropic_service: Optional[AnthropicService] = None,
        openrouter_service: Optional[OpenRouterService] = None,
    ):
        """
        Initialize the LLM router.

        Args:
            anthropic_service: Anthropic service instance (created if None)
            openrouter_service: OpenRouter service instance (created if None)
        """
        self.anthropic = anthropic_service or AnthropicService()
        self.openrouter = openrouter_service or OpenRouterService()

        self.health = {
            "anthropic": ProviderHealth("anthropic"),
            "openrouter": ProviderHealth("openrouter"),
        }

        self.preferred_provider = "anthropic"
        self.force_provider: Optional[str] = None

    async def _execute_with_health_tracking(
        self,
        provider_name: str,
        coro,
    ):
        """Execute a coroutine and track health."""
        start_time = datetime.utcnow()

        try:
            result = await coro
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            self.health[provider_name].record_success(latency_ms)
            return result

        except Exception as e:
            error_msg = str(e)
            self.health[provider_name].record_failure(error_msg)
            raise

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Send a chat request with automatic failover.

        Args:
            request: Chat request

        Returns:
            ChatResponse

        Raises:
            Exception: If all providers fail
        """
        # Determine provider order
        provider_order = self._get_provider_order(request)

        last_error = None

        for provider_name in provider_order:
            health = self.health[provider_name]

            if not health.available and not self.force_provider:
                continue

            try:
                if provider_name == "anthropic":
                    service = self.anthropic
                else:
                    service = self.openrouter

                result = await self._execute_with_health_tracking(
                    provider_name,
                    service.chat(request),
                )
                return result

            except Exception as e:
                last_error = e
                continue

        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")

    async def chat_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Send a streaming chat request with automatic failover.

        Note: Failover only works at connection time, not mid-stream.

        Args:
            request: Chat request

        Yields:
            StreamingChunk events

        Raises:
            Exception: If all providers fail
        """
        provider_order = self._get_provider_order(request)

        last_error = None

        for provider_name in provider_order:
            health = self.health[provider_name]

            if not health.available and not self.force_provider:
                continue

            try:
                if provider_name == "anthropic":
                    service = self.anthropic
                else:
                    service = self.openrouter

                # Note: We can't track health mid-stream easily
                # The health will be tracked on the initial connection
                async for chunk in service.chat_stream(request):
                    yield chunk
                return

            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All providers failed. Last error: {last_error}")

    def _get_provider_order(self, request: ChatRequest) -> List[str]:
        """Get ordered list of providers to try."""
        # Check if request specifies fallback
        use_fallback = (
            request.provider and request.provider.use_fallback
        )

        # Build provider list
        providers = ["anthropic"]
        if use_fallback or self.force_provider == "openrouter":
            providers.append("openrouter")

        return providers

    async def get_provider_status(self) -> List[ProviderStatus]:
        """
        Get status of all providers.

        Returns:
            List of ProviderStatus for each provider
        """
        # Get models for each provider
        anthropic_models = await self.anthropic.list_models()
        openrouter_models = await self.openrouter.list_models()

        return [
            self.health["anthropic"].to_status(anthropic_models),
            self.health["openrouter"].to_status(openrouter_models),
        ]

    async def health_check(self) -> Dict[str, bool]:
        """
        Run health checks on all providers.

        Returns:
            Dict mapping provider name to health status
        """
        results = {}

        # Check Anthropic
        try:
            anthropic_healthy = await asyncio.wait_for(
                self.anthropic.health_check(),
                timeout=10.0,
            )
            if anthropic_healthy:
                self.health["anthropic"].record_success(100)
            else:
                self.health["anthropic"].record_failure("Health check failed")
            results["anthropic"] = anthropic_healthy
        except Exception as e:
            self.health["anthropic"].record_failure(str(e))
            results["anthropic"] = False

        # Check OpenRouter
        try:
            openrouter_healthy = await asyncio.wait_for(
                self.openrouter.health_check(),
                timeout=10.0,
            )
            if openrouter_healthy:
                self.health["openrouter"].record_success(100)
            else:
                self.health["openrouter"].record_failure("Health check failed")
            results["openrouter"] = openrouter_healthy
        except Exception as e:
            self.health["openrouter"].record_failure(str(e))
            results["openrouter"] = False

        return results

    def set_preferred_provider(self, provider: str):
        """Set preferred provider."""
        if provider in self.health:
            self.preferred_provider = provider
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def force_provider_switch(self, provider: str):
        """Force all requests to use a specific provider."""
        if provider in self.health:
            self.force_provider = provider
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def clear_force_provider(self):
        """Clear forced provider and use automatic selection."""
        self.force_provider = None

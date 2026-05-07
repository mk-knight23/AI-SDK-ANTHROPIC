# Current Feature Inventory

## Repository

- Name: `AI-SDK-ANTHROPIC`
- SDK: Anthropic Claude SDK
- Positioning: Claude-centered secure agent runtime with policy-first design.

## Implemented Today

- Mission-to-agent routing through the shared Agents Army registry.
- FastAPI service with `GET /health` and `POST /run`.
- CLI runner for local mission smoke checks.
- Anthropic client initialization path with graceful dependency handling.
- Dockerfile, GitHub Actions CI, pytest contract tests, and repository metadata.

## Not Yet Implemented

- Connect routed missions to Claude Messages API calls.
- Add safety policy templates and refusal/audit logging.
- Add evaluation traces for cost, latency, and policy compliance.

## Verification Contract

- The local runner must complete without crashing when optional SDK credentials are missing.
- The API contract must return routing and verification fields.
- Tests must prove mission routing and a security-focused SENTINEL route.

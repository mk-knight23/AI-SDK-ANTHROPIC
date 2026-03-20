# Project Brain: anthropic

## Purpose
Implement Kazi's Agents Army mission routing on the anthropic framework.

## Core Mechanism
- Receive mission text.
- Route to primary + support agents via shared core router.
- Convert routed mission into framework-native execution primitives.

## Current State
- Adapter skeleton implemented with optional imports.
- Demo mission execution path available via `python runner.py`.

## Production Plan
- Configure provider keys via environment variables.
- Add persistent memory/checkpoint backends where available.
- Add eval, tracing, and latency/cost dashboards.
- Add CI tests and deployment workflows.

## Risks
- Framework API drift across versions.
- Inference cost growth under multi-agent fan-out.
- Tool execution needs policy and approval guardrails.

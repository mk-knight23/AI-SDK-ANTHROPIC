# AI-SDK-ANTHROPIC

Claude-centered secure agent runtime with policy-first design.

## What this adapter does

This repository is one runtime adapter in Kazi's AI SDK ecosystem. It takes a mission, routes it through the shared Agents Army skill registry, and validates how the Anthropic SDK should participate in the larger control plane.

## Required skill set

- Claude API operations
- policy design
- safety review
- prompt governance
- Mission routing and support-agent selection
- Contract testing and release verification
- Secure environment-variable based secret handling

## Local run

```bash
python3 runner.py --mission "build secure api and deploy"
```

## API run

```bash
python3 -m pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Test

```bash
python3 -m pytest
```

## Ecosystem contract

- Input: `{ "mission": "..." }`
- Output includes `primary`, `support`, and `verification`
- Provider SDK setup is lazy: missing provider credentials or optional SDKs return a verification note instead of crashing the control plane
- Secrets must be supplied through environment variables, never committed

## Portfolio positioning

This adapter demonstrates Anthropic expertise inside a larger platform story: routing, orchestration, safety, verification, and deployment discipline across multiple AI frameworks.

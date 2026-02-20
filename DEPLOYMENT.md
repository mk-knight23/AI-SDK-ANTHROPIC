# Deployment Guide

## Fly.io Deployment

### Prerequisites

1. Install Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login to Fly:
```bash
fly auth login
```

### Deploy Backend

```bash
# Create the app (first time only)
fly apps create research-synthesis-api

# Deploy
fly deploy --config fly.backend.toml
```

### Deploy Frontend

```bash
# Create the app (first time only)
fly apps create research-synthesis-web

# Set secrets (if needed)
fly secrets set API_URL=https://research-synthesis-api.fly.dev --app research-synthesis-web

# Deploy
fly deploy --config fly.frontend.toml
```

### Verify Deployment

```bash
# Check backend health
curl https://research-synthesis-api.fly.dev/health

# Open frontend
fly open --app research-synthesis-web
```

### Monitoring

```bash
# View logs
fly logs --app research-synthesis-api
fly logs --app research-synthesis-web

# View status
fly status --app research-synthesis-api
fly status --app research-synthesis-web
```

### Scaling

```bash
# Scale to 2 machines
fly scale count 2 --app research-synthesis-api

# Scale memory
fly scale memory 1024 --app research-synthesis-api
```

# anthropic Adapter

## CLI Run
```bash
python runner.py --mission "your mission"
```

## API Run
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Health Check
```bash
curl http://localhost:8000/health
```

## Mission Invoke
```bash
curl -X POST http://localhost:8000/run \
  -H "content-type: application/json" \
  -d '{"mission":"build secure api and deploy"}'
```

## Install
```bash
pip install -r requirements.txt
```

## Deploy
- Build Docker image from included Dockerfile.
- Provide secrets via environment variables.

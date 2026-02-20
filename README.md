# ResearchSynthesis

AI-powered research synthesis platform combining academic paper analysis with intelligent summarization.

## Tech Stack

![Remix](https://img.shields.io/badge/Remix-000000?style=for-the-badge&logo=remix&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## Project Structure

```
.
├── frontend/          # Remix frontend application
├── backend/           # Python FastAPI backend
├── .github/           # GitHub Actions workflows
├── docker-compose.yml # Docker Compose configuration
└── README.md          # This file
```

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.12+
- Docker (optional)

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd projects/07-research-synthesis

# Start the backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# In a new terminal, start the frontend
cd frontend
npm install
npm start
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "0.1.0",
  "service": "research-synthesis-api"
}
```

## Testing

```bash
# Backend tests
cd backend
pytest -v

# Frontend tests
cd frontend
npm test
```

## Deployment

### Fly.io Deployment

1. Install Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login to Fly:
```bash
fly auth login
```

3. Launch the application:
```bash
# Create fly.toml for backend
fly launch --name research-synthesis-api --dockerfile backend/Dockerfile

# Deploy backend
fly deploy --config fly.backend.toml

# Deploy frontend
fly deploy --config fly.frontend.toml
```

### Environment Variables

Create a `.env` file in the backend directory:

```env
PORT=8000
DEBUG=false
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

MIT

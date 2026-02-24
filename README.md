🟣 Anthropic Claude SDK

[![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)]()
[![Version](https://img.shields.io/badge/Version-0.1.0-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-80%25-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-fly-io-informational.svg)](https://fly.io/)

## 📋 Overview

Anthropic Claude SDK integration with Next.js and FastAPI

**Current Version:** 0.1.0 | **Status:** Beta | **Completion:** 75%

---

## 📖 About The Project

### What is Anthropic Claude SDK?

A production-ready AI SDK application showcasing Anthropic's powerful capabilities for building integrating Claude models. This project demonstrates enterprise-grade patterns for AI-native application development.

### Why This Project Exists?

The AI SDK landscape is rapidly evolving with powerful frameworks emerging for building AI-native applications. This project serves as:

- **Reference Implementation**: Production-ready code patterns for Anthropic applications
- **Learning Resource**: Best practices for integrating AI capabilities into web applications
- **Starting Point**: Solid foundation for building your own AI-powered products
- **Comparison Tool**: Compare different AI SDK approaches across our 10-project ecosystem

### Problem Statement

Building AI applications requires integrating multiple complex systems:
- LLM API management and rate limiting
- Context persistence and memory management
- Vector database operations for RAG
- Real-time streaming responses
- prompt engineering and Claude features

This project provides a complete, working solution to these challenges.

### Key Features

✨ **Anthropic Integration**
- Full-featured Anthropic SDK
- Claude 3.5 Sonnet & Opus support
- Extended context windows (200K tokens)
- Prompt engineering best practices
- Streaming responses
- Vision capabilities
- MCP (Model Context Protocol) support

🎨 **Modern UI**
- Modern React-based frontend with TypeScript
- Responsive design with dark mode
- Real-time updates via WebSocket
- Accessible components (WCAG 2.1 AA)

🚀 **High Performance**
- Optimistic UI updates
- Request deduplication
- Response caching with Redis
- Database connection pooling

🔒 **Security First**
- API key encryption at rest
- Rate limiting per user
- Input sanitization
- CORS protection

📊 **Observability**
- Structured logging
- Performance metrics
- Error tracking
- Usage analytics

🧪 **Well Tested**
- 80%+ test coverage
- Unit, integration, and E2E tests
- Load testing scripts
- Test data fixtures

---

## 🛠️ Tech Stack


### Frontend

| Technology | Version | Description |
|------------|---------|-------------|
| [![Next.js](https://nextjs.org/)](https://img.shields.io/badge/Next.js-15.1.0-61DAFB.svg) | 15 | Next.js frontend framework |
| [![React](https://react.dev/)](https://img.shields.io/badge/React-19.0-61DAFB.svg) | 19 | React frontend framework |
| [![TypeScript](https://www.typescriptlang.org/)](https://img.shields.io/badge/TypeScript-5-3178C6.svg) | 5 | Type-safe JavaScript |

### Backend

| Technology | Version | Description |
|------------|---------|-------------|
| [![FastAPI](https://fastapi.tiangolo.com/)](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg) | 0.115.0 | FastAPI backend/framework |
| [![Anthropic](https://github.com/anthropics/anthropic-sdk-python)](https://img.shields.io/badge/Anthropic-Latest-red.svg) | SDK | Anthropic backend/framework |
| [![Python](https://www.python.org/)](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg) | 3.12+ | Python backend/framework |

### Deployment

| Platform | Purpose |
|----------|---------|
| [![Fly.io](https://img.shields.io/badge/Deploy-Fly.io-informational.svg)](https://fly.io/) | Primary deployment platform |
| [![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg)](https://docker.com/) | docker |
| [![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)](https://github.com/features/actions) | github actions |

---

## 📊 Current Stage

### Development Status: Beta

**Completion: 75%**

#### ✅ Completed Features

- Core Anthropic integration
- Modern frontend with TypeScript
- Backend API framework
- PostgreSQL database setup
- Redis caching layer
- Authentication system
- WebSocket for real-time updates
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite

#### 🚧 Known Limitations

- Rate limiting is basic (needs token-bucket algorithm)
- Vector store supports only basic operations
- No advanced caching strategies
- Limited monitoring/alerting
- E2E tests need expansion

#### 🗺️ Roadmap

**Q1 2025: Core Enhancement**
- [ ] Advanced caching strategies
- [ ] Enhanced rate limiting
- [ ] Performance optimization
- [ ] Expanded test coverage (90%+)

**Q2 2025: Enterprise Features**
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Custom model fine-tuning integration
- [ ] Plugin system

**Q3 2025: Ecosystem**
- [ ] Additional model provider integrations
- [ ] Advanced agent orchestration
- [ ] Workflow automation builder
- [ ] Mobile SDK

**Contributions Welcome!** See [Contributing](#contributing) section.

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** 20+ and npm 10+
- **Python** 3.12+ (for backend)
- **PostgreSQL** 15+ (or use Docker)
- **Redis** 7+ (or use Docker)
- **Git** for version control

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-org/AI-SDK-ANTHROPIC.git
cd AI-SDK-ANTHROPIC
```

2. **Install Frontend Dependencies**

```bash
cd frontend
npm install
```

3. **Install Backend Dependencies**

```bash
cd ../backend
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

4. **Environment Setup**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
# Required variables:
# - OPENAI_API_KEY or ANTHROPIC_API_KEY
# - DATABASE_URL
# - REDIS_URL
# - JWT_SECRET
```

5. **Start PostgreSQL and Redis**

```bash
# Using Docker Compose (recommended)
docker-compose up -d postgres redis

# Or run locally
# See docker-compose.yml for configuration
```

6. **Initialize Database**

```bash
cd backend
# Run migrations
alembic upgrade head

# Or create tables
python -c "from app.database import init_db; init_db()"
```

7. **Start the Application**

```bash
# Terminal 1: Backend
cd backend
python main.py
# Backend runs on http://localhost:8000

# Terminal 2: Frontend
cd frontend
npm run dev
# Frontend runs on http://localhost:3000
```

8. **Verify Installation**

Open your browser:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

### Docker Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📁 Project Structure

```
AI-SDK-ANTHROPIC/
  CLAUDE.md
  CONTRIBUTING.md
  LICENSE
  README.md
  frontend/
    src/
      components/
      lib/
      app/
  backend/
    app/
      api/
      models/
      services/
    tests/
  docs/
```

### Key Files and Directories

| Path | Purpose |
|------|---------|
| `frontend/src/` | Frontend application code |
| `frontend/components/` | Reusable UI components |
| `frontend/lib/` | Utility functions and helpers |
| `backend/app/` | Backend application code |
| `backend/app/api/` | API route handlers |
| `backend/app/models/` | Database models |
| `backend/app/services/` | Business logic layer |
| `backend/tests/` | Test suite |
| `docs/` | Additional documentation |
| `docker-compose.yml` | Docker orchestration |
| `.github/workflows/` | CI/CD pipelines |

---

## 🏗️ Architecture


```mermaid
graph TB
    subgraph "Client Layer"
        A[User Browser]
        B[Frontend]
    end

    subgraph "API Layer"
        C[API Gateway]
        D[Backend API]
        E[Authentication]
    end

    subgraph "AI Layer"
        F[Anthropic SDK]
        G[LLM Provider]
        H[Vector Store]
        I[Memory Store]
    end

    subgraph "Data Layer"
        J[(PostgreSQL)]
        K[Redis Cache]
        L[File Storage]
    end

    A -->|HTTP/WebSocket| B
    B -->|REST API| C
    C --> D
    D --> E
    D --> F
    F -->|API Calls| G
    F -->|Embeddings| H
    F -->|Context| I
    D --> J
    D --> K
    D --> L

    style F fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    style D fill:#4ecdc4,stroke:#087f5b,stroke-width:2px
    style B fill:#74c0fc,stroke:#1971c2,stroke-width:2px
```

**Architecture Overview:**

1. **Client Layer**: Modern frontend provides responsive UI with real-time updates
2. **API Layer**: Backend framework handles business logic and request routing
3. **AI Layer**: Anthropic orchestrates AI model interactions and workflows
4. **Data Layer**: Persistent storage with caching for optimal performance

**Data Flow:**
- User actions flow from Frontend → Backend → AI SDK
- AI SDK processes requests through LLM providers with context from vector stores
- Responses stream back through WebSocket for real-time updates
- All interactions logged to database for analytics and debugging

---

## 📚 API Documentation


### Authentication

All API endpoints require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.example.com/v1/chat
```

### Available Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/chat/completions` | Generate chat completion | Yes |
| POST | `/api/v1/completions` | Text completion | Yes |
| POST | `/api/v1/embeddings` | Generate embeddings | Yes |
| GET | `/api/v1/models` | List available models | Yes |
| POST | `/api/v1/agents/run` | Execute agent task | Yes |
| GET | `/api/v1/agents/{id}` | Get agent details | Yes |
| POST | `/api/v1/agents/{id}/message` | Send message to agent | Yes |
| GET | `/api/v1/threads/{id}` | Get conversation thread | Yes |
| POST | `/api/v1/threads` | Create new thread | Yes |
| DELETE | `/api/v1/threads/{id}` | Delete thread | Yes |
| GET | `/api/v1/health` | Health check | No |
| GET | `/api/v1/metrics` | System metrics | Yes |

### Request/Response Examples

#### Chat Completion

**Request:**
```bash
curl -X POST https://api.example.com/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking!"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

#### Streaming Response

**Request:**
```bash
curl -X POST https://api.example.com/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

**Response (Server-Sent Events):**
```
data: {"id": "cmpl-1", "choices": [{"delta": {"content": "Once"}]}
data: {"id": "cmpl-2", "choices": [{"delta": {"content": " upon"}]}
data: {"id": "cmpl-3", "choices": [{"delta": {"content": " a"}]}
data: [DONE]
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request body format |
| 401 | Unauthorized | Verify API key is valid |
| 429 | Rate Limited | Implement exponential backoff |
| 500 | Internal Error | Check server logs, retry later |
| 503 | Service Unavailable | Service temporarily down, retry |

---

## 🏷️ Tags & Badges Used

### Git Tags Convention

We use semantic versioning for git tags:

```bash
# Format: v<major>.<minor>.<patch>
v0.1.0    # Initial release
v0.1.1    # Patch release (bug fixes)
v0.2.0    # Minor release (new features)
v1.0.0    # Major release (breaking changes)
```

### Text Tags in Documentation

| Tag | Usage |
|-----|-------|
| ✅ | Completed features |
| 🚧 | Work in progress |
| 📋 | Planned features |
| ⚠️ | Warnings/limitations |
| 💡 | Tips and suggestions |
| 🔒 | Security notes |
| 🚀 | Performance notes |
| 📊 | Metrics/monitoring |

### Release Naming

Release names follow the pattern: **[SDK Name] [Version]**

Example: "Anthropic Latest"

### Versioning Strategy

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality
- **PATCH**: Backwards-compatible bug fixes

---

## 🧪 Testing


### Test Framework

- **Frontend**: Jest + React Testing Library
- **Backend**: pytest + pytest-asyncio
- **E2E**: Playwright

### Running Tests

```bash
# Frontend tests
cd frontend
npm test                    # Run all tests
npm run test:watch          # Watch mode
npm run test:coverage       # With coverage report

# Backend tests
cd backend
pytest                      # Run all tests
pytest -v                   # Verbose output
pytest --cov=app            # With coverage
pytest -k "test_agent"      # Run specific tests

# E2E tests
npm run test:e2e            # Run E2E tests
```

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── components/         # Component tests
│   ├── services/           # Service tests
│   └── utils/              # Utility tests
├── integration/            # Integration tests
│   ├── api/                # API endpoint tests
│   └── database/           # Database tests
└── e2e/                    # End-to-end tests
    ├── flows/              # User flow tests
    └── scenarios/          # Scenario tests
```

### Current Coverage

- **Frontend**: 80%
- **Backend**: 85%
- **Overall**: 82%

### Testing Documentation

📄 [Full Testing Guide](docs/TESTING.md)

---

## 📄 Documentation

### Available Documentation

| Document | Description | Link |
|----------|-------------|------|
| API Reference | Complete API documentation | [API Docs](docs/API.md) |
| Deployment Guide | Production deployment instructions | [Deployment Guide](docs/DEPLOYMENT.md) |
| Testing Guide | Comprehensive testing documentation | [Testing Guide](docs/TESTING.md) |
| Contributing Guide | How to contribute to the project | [Contributing Guide](CONTRIBUTING.md) |
| Architecture | Deep dive into system architecture | [Architecture Docs](docs/ARCHITECTURE.md) |
| CHANGELOG | Version history and changes | [CHANGELOG.md](CHANGELOG.md) |

### Quick Links

- 📖 [Project Wiki](https://github.com/your-org/AI-SDK-ANTHROPIC/wiki)
- 💬 [Discussions](https://github.com/your-org/AI-SDK-ANTHROPIC/discussions)
- 🐛 [Issue Tracker](https://github.com/your-org/AI-SDK-ANTHROPIC/issues)
- 📢 [Release Notes](https://github.com/your-org/AI-SDK-ANTHROPIC/releases)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# API Keys (Required)
OPENAI_API_KEY=sk-...                    # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...             # Anthropic API key

# Database (Required)
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
REDIS_URL=redis://localhost:6379/0

# Application (Required)
JWT_SECRET=your-secret-key-here
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000

# Optional
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
MAX_TOKENS=4096                          # Max tokens per request
TEMPERATURE=0.7                          # Default temperature
RATE_LIMIT_PER_MINUTE=60                 # Rate limit per user

# Feature Flags
ENABLE_STREAMING=true
ENABLE_WEBSOCKET=true
ENABLE_ANALYTICS=false
```

### Config Files

| File | Location | Purpose |
|------|----------|---------|
| `next.config.js` | `frontend/` | Next.js configuration |
| `vite.config.ts` | `frontend/` | Vite configuration |
| `tsconfig.json` | `frontend/` | TypeScript configuration |
| `tailwind.config.js` | `frontend/` | Tailwind CSS configuration |
| `pytest.ini` | `backend/` | Pytest configuration |
| `alembic.ini` | `backend/` | Database migration config |
| `docker-compose.yml` | Root | Docker services config |

### Default Values

See `.env.example` for complete list with default values.

---

## 🔄 How To Upgrade


### Dependency Updates

```bash
# Frontend dependencies
cd frontend
npm update                  # Update dependencies
npm outdated                # Check for updates
npm install package@latest  # Update specific package

# Backend dependencies
cd backend
pip install --upgrade -r requirements.txt
pip list --outdated        # Check for updates
```

### Migration Guides

#### Version 0.1.0 → 0.2.0 (Planned)

- [ ] Update AI SDK to latest version
- [ ] Migrate to new API format
- [ ] Update authentication flow
- [ ] Review breaking changes

### Breaking Changes

**Current Version: 0.1.0**

No breaking changes in current version.

### Version Upgrade Path

1. **Check release notes**: Review [`CHANGELOG.md`](CHANGELOG.md) for changes
2. **Update dependencies**: Run update commands above
3. **Run tests**: Ensure all tests pass
4. **Test locally**: Verify functionality in development
5. **Deploy**: Use CI/CD pipeline for safe deployment

### Rollback Procedure

If issues occur after upgrade:

```bash
git checkout previous-version
npm install  # or pip install -r requirements.txt
npm run build
# Deploy previous version
```

---

## 🗺️ Future Roadmap

### Planned Features

#### 🎯 High Priority

- [ ] **Advanced Caching**: Implement multi-layer caching strategy
- [ ] **Rate Limiting**: Token-bucket algorithm with Redis
- [ ] **Monitoring**: Prometheus + Grafana dashboards
- [ ] **Testing**: Increase coverage to 90%+

#### 🔮 Medium Priority

- [ ] **Multi-tenancy**: Support for multiple organizations
- [ ] **Analytics**: Usage analytics and cost tracking
- [ ] **Fine-tuning**: Custom model fine-tuning integration
- [ ] **Plugin System**: Extensible plugin architecture

#### 💡 Lower Priority

- [ ] **Mobile Apps**: React Native mobile application
- [ ] **Desktop App**: Electron desktop application
- [ ] **CLI Tool**: Command-line interface for API
- [ ] **GraphQL API**: Alternative to REST API

### Technical Debt

- [ ] Refactor: Improve error handling consistency
- [ ] Refactor: Extract common utilities to shared package
- [ ] Tests: Add more integration tests
- [ ] Docs: Improve code documentation coverage
- [ ] Performance: Optimize database queries
- [ ] Security: Implement request signing

### Investment Areas

We're particularly interested in:

1. **Performance**: Database query optimization, caching strategies
2. **Security**: Enhanced authentication, audit logging
3. **Developer Experience**: Better CLI tools, local development setup
4. **Observability**: Distributed tracing, error aggregation

### Community Contributions Welcome!

We welcome contributions in all areas. See [Contributing](#contributing) below.

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "Module not found" errors

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install

cd ../backend
pip install --force-reinstall -r requirements.txt
```

#### Issue: Database connection refused

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Start PostgreSQL
docker-compose up -d postgres

# Check connection string in .env
echo $DATABASE_URL
```

#### Issue: API rate limiting errors

**Solution:**
- Implement exponential backoff in your client
- Check your API key quota
- Consider upgrading your API plan

#### Issue: WebSocket connection fails

**Solution:**
```bash
# Check WebSocket URL in .env
WEBSOCKET_URL=ws://localhost:8000/ws

# Verify WebSocket is enabled
curl -I http://localhost:8000/docs
```

#### Issue: Tests failing locally

**Solution:**
```bash
# Ensure test environment is set up
cd backend
pytest --create-test-db

# Run with verbose output
pytest -vvs

# Run specific test
pytest tests/test_api.py::test_create_agent
```

### Getting Help

If you're still stuck:

1. 📖 Check the [Documentation](#-documentation)
2. 🔍 Search [existing issues](https://github.com/your-org/AI-SDK-ANTHROPIC/issues)
3. 💬 Start a [Discussion](https://github.com/your-org/AI-SDK-ANTHROPIC/discussions)
4. 🐛 [Create an issue](https://github.com/your-org/AI-SDK-ANTHROPIC/issues/new) with details

### Debug Mode

Enable debug logging:

```bash
# In .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

---

## 🤝 Contributing

We love contributions! Whether you're fixing a bug, adding a feature, or improving documentation.

### How to Contribute

1. **Fork the repository**

```bash
# Click "Fork" on GitHub, then clone your fork
git clone https://github.com/your-username/AI-SDK-ANTHROPIC.git
cd AI-SDK-ANTHROPIC
```

2. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
# Or for bug fixes
git checkout -b fix/your-bug-fix
```

3. **Make your changes**

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed
- Follow our [Code Style Guidelines](docs/CODE_STYLE.md)

4. **Run tests**

```bash
cd frontend && npm test
cd ../backend && pytest
```

5. **Commit your changes**

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

6. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

7. **Create Pull Request**

- Go to GitHub and create a Pull Request
- Describe your changes clearly
- Link any related issues
- Wait for review

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/AI-SDK-ANTHROPIC.git
cd AI-SDK-ANTHROPIC

# Install dependencies
cd frontend && npm install
cd ../backend && pip install -r requirements.txt

# Setup pre-commit hooks
npm run install:pre-commit-hooks

# Start development servers
npm run dev:all  # Starts both frontend and backend
```

### Code Style

- **Frontend**: ESLint + Prettier (configured in `.eslintrc`)
- **Backend**: Black + isort (configured in `pyproject.toml`)
- **Commit**: Conventional Commits + Commitlint

Run linting:
```bash
npm run lint         # Frontend
npm run format       # Auto-format code
```

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new features
4. Keep PRs focused and atomic
5. Request review from maintainers
6. Address review feedback

### Review Guidelines

- PRs reviewed within 48 hours
- At least one approval required
- All CI checks must pass
- No merge conflicts

---

## 📜 License & Authors

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 AI SDK Projects

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Authors

**Core Team**
- **Your Name** - Project Lead - [@yourhandle](https://github.com/yourhandle)
- **Contributor Name** - Backend Developer - [@contributor](https://github.com/contributor)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full list of contributors.

### Acknowledgments

- **Anthropic Team** - For the amazing AI SDK
- **React Team** - For the excellent frontend framework
- **FastAPI/Django/Flask Team** - For the robust backend framework
- **Open Source Community** - For inspiration and feedback

### Related Projects

This is part of the AI SDK Projects ecosystem - 10 projects showcasing different AI SDKs:

- [🦜 LangChain SDK](../AI-SDK-LANGCHAIN)
- [👥 CrewAI SDK](../AI-SDK-CREWAI)
- [🕸️ LangGraph SDK](../AI-SDK-LANGGRAPH)
- [🤖 AutoGen SDK](../AI-SDK-AUTOGEN)
- [🔷 OpenAI SDK](../AI-SDK-OPENAI)
- [🟣 Anthropic SDK](../AI-SDK-ANTHROPIC)
- [▲ Vercel AI SDK](../AI-SDK-VERCEL-AI)
- [🌾 Haystack SDK](../AI-SDK-HAYSTACK)
- [🦙 LlamaIndex SDK](../AI-SDK-LAMA-INDEX)
- [🧠 Semantic Kernel SDK](../AI-SDK-SEMANTIC-KERNEL)

---

## 📞 Contact & Support

- **Website**: https://ai-sdk-projects.example.com
- **Documentation**: https://docs.ai-sdk-projects.example.com
- **Twitter**: [@ai_sdk_projects](https://twitter.com/ai_sdk_projects)
- **Discord**: [Join our Discord](https://discord.gg/ai-sdk-projects)
- **Email**: support@ai-sdk-projects.example.com

---

<div align="center">

**⭐ Star us on GitHub** — it helps!

**🦸 Sponsor Us** — Support our open-source work

Made with ❤️ by the AI SDK Projects Team

</div>

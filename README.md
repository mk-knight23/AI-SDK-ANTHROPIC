# AI-SDK-ANTHROPIC

[![AI-SDK Ecosystem](https://img.shields.io/badge/AI--SDK-ECOSYSTEM-part%20-of-blue)](https://github.com/mk-knight23/AI-SDK-ECOSYSTEM)
[![Anthropic](https://img.shields.io/badge/Anthropic-1.42.0-713f99b)](https://www.anthropic.com/)
[![Remix](https://img.shields.io/badge/Remix-2.0-blue)](https://remix.run/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)

> **Framework**: Anthropic Claude (Extended Thinking + Computer Use)
> **Stack**: Remix + FastAPI

---

## 🎯 Project Overview

**AI-SDK-ANTHROPIC** showcases the Anthropic Claude API with extended thinking, message streaming, and computer use capabilities. It demonstrates Claude's 200K context window, artifact generation, and advanced reasoning for building sophisticated AI applications.

### Key Features

- 🧠 **Extended Thinking** - Claude's chain-of-thought reasoning
- 💬 **Message Streaming** - Real-time token-by-token responses
- 🖥️ **Computer Use** - Claude can control computers via API
- 📄 **Artifact Generation** - PDF, code, and content creation
- 🔗 **OpenRouter Fallback** - Multi-provider resilience

---

## 🛠 Tech Stack

| Technology | Purpose |
|-------------|---------|
| Remix | Full-stack framework |
| FastAPI | Backend API |
| Anthropic SDK | Claude integration |
| Chakra UI | Components |
| GraphQL | API layer |

---

## 🚀 Quick Start

```bash
# Frontend
cd app && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && python main.py
```

---

## 🔌 API Integrations

| Provider | Usage |
|----------|-------|
| Anthropic Claude | Primary (extended thinking) |
| OpenRouter | Fallback router |

---

## 📦 Deployment

**Fly.io** (both frontend and backend)

```bash
fly deploy
```

---

## 📁 Project Structure

```
AI-SDK-ANTHROPIC/
├── app/              # Remix application
├── backend/          # FastAPI backend
└── README.md
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Part of the [AI-SDK Ecosystem](https://github.com/mk-knight23/AI-SDK-ECOSYSTEM)**

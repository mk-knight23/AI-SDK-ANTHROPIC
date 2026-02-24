# Testing Guide

**AI-SDK-ANTHROPIC** - Comprehensive Testing Documentation

---

## Testing Philosophy

This project follows **Test-Driven Development (TDD)**:

1. **RED** - Write failing test first
2. **GREEN** - Write minimal implementation to pass
3. **IMPROVE** - Refactor while keeping tests green

**Target Coverage**: 80%+ across all modules

---

## Running Tests

### Backend Tests

```bash
cd backend
pytest --cov=app --cov-report=html
```

### Frontend Tests

```bash
cd frontend
npm test -- --coverage
```

---

## Test Structure

```
backend/
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/              # End-to-end tests

frontend/
├── __tests__/
│   ├── unit/             # Component tests
│   └── e2e/             # E2E tests
```

---

## Coverage

**Target**: 80%+ coverage

View coverage reports:
- Backend: `htmlcov/index.html`
- Frontend: `coverage/lcov-report/index.html`

---

For more details, see the project README.

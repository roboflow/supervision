# GitHub Copilot Instructions for Supervision

This file provides context-aware guidance for GitHub Copilot when working in the Supervision repository.

---

## 📚 Repository Overview

**Supervision** is a Python library providing reusable computer vision utilities for working with object detection models (YOLO, SAM, etc.). It offers tools for detections processing, tracking, annotation, and dataset management.

- **Languages**: Python 3.9+
- **Key Dependencies**: NumPy, OpenCV, SciPy
- **License**: MIT

---

## 🏗️ Project Structure

```
supervision/
├── src/
│   └── supervision/     # Main library code
│       ├── detection/   # Detection utilities
│       ├── draw/        # Annotation and visualization
│       ├── tracker/     # Object tracking
│       ├── dataset/     # Dataset management
│       └── utils/       # Shared utilities
├── tests/               # Test suite (mirrors src/supervision/)
├── docs/                # MkDocs documentation
└── examples/            # Usage examples
```

---

## 🔧 Development Commands

**Setup:**

```bash
# Install dependencies
uv sync --group dev --group docs --extra metrics

# Install pre-commit hooks
uv run pre-commit install
```

**Quality Checks:**

```bash
# Run all pre-commit hooks (formatting, linting, type checking)
uv run pre-commit run --all-files

# Run tests with coverage
uv run pytest --cov=supervision
```

**Documentation:**

```bash
# Serve docs locally at http://127.0.0.1:8000
uv run mkdocs serve
```

---

## 💻 Code Conventions

### General Guidelines

- Follow **[AGENTS.md](../AGENTS.md)** for task-based development workflows
- Reference **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed contribution guidelines
- All code must pass `pre-commit` hooks before committing

### Code Style

- **Formatting**: Enforced by `ruff-format`, `prettier` (pre-commit)
- **Linting**: Enforced by `ruff-check` (pre-commit)
- **Type Hints**: Required on all new code
- **Docstrings**: Required using [Google Python style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
    - Must include usage examples with primitive values
    - Serve as runnable documentation

### Performance

- Avoid unnecessary NumPy array copies
- Prefer vectorized operations over Python loops
- Use OpenCV operations efficiently

### API Design

- Follow existing naming patterns for consistency
- Maintain backward compatibility unless explicitly breaking
- Prefer functional utilities over complex classes

---

## 🧪 Testing Requirements

All new features must include:

- Unit tests covering happy path and edge cases
- Tests for `None`, empty inputs, large arrays, boundary conditions
- Clear test names describing what they validate
- Proper assertions (not just "no exception raised")

---

## 📝 Documentation Requirements

For new public functions/classes:

- Google-style docstrings with parameters, returns, exceptions
- Usage examples in docstrings
- Entry in appropriate `docs/*.md` file
- Reference in `mkdocs.yml` navigation

---

## 🔍 Pull Request Reviews

**When reviewing PRs, follow the comprehensive [PR Review Guidelines](CONTRIBUTING.md#pr-review-guidelines).**

Quick checklist:

- Tests included and passing
- Docstrings follow Google style with examples
- Pre-commit hooks pass
- Breaking changes documented
- Score code quality, testing, docs (n/5 scale)
- Use inline comments + GitHub suggestion format

---

## 🌿 Branching & Commits

- Branch from `develop` using prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `perf/`, `test/`, `chore/`
- Use **conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`
- All PRs target `develop` branch

---

## 🎯 Context-Aware Behavior

**For general development tasks**: Follow [AGENTS.md](../AGENTS.md)
**For pull request reviews**: Follow [PR Review Guidelines](CONTRIBUTING.md#pr-review-guidelines)
**For detailed processes**: Consult [CONTRIBUTING.md](CONTRIBUTING.md)

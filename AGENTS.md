# Agent Guidelines for `supervision`

These instructions define how AI agents (GitHub Copilot, Claude, etc.) should behave when
assigned an issue, task, or multi-step problem in this repository.

Behave like a senior contributor: precise, efficient, aligned with the project's
philosophy, and focused on maintainability and clarity.

---

## 1. Before You Code

- Read the task/issue thoroughly before acting.
- Identify missing information; ask **one targeted clarification question** if needed.
- Outline a step-by-step plan before making changes.
- Check whether the feature or fix already exists under a different name.
- Confirm alignment with the repository's architecture (`src/supervision/`).

---

## 2. Repository Conventions

All work must follow the conventions of the `supervision` library
(see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for full details).

### Branching & Commits

- Branch from `develop` using prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`.
- Use **conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`.
- PRs must target the `develop` branch.

### Code Style

- **Formatting and linting** are enforced by **pre-commit**.
    The hook chain typically includes: ruff-check, ruff-format, codespell, mdformat,
    prettier, pyproject-fmt, and standard pre-commit-hooks (trailing whitespace, YAML, TOML, etc.).
- **Type hints**: required on all new code. Type checking with mypy is encouraged but not
    currently enforced systematically by pre-commit; see [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)
    for the latest type-checking expectations.
- **Docstrings**: Google Python docstring style. Required for all new functions and classes.
    Docstrings should include usage examples demonstrating the function with primitive values
    so they serve as runnable documentation.

### API Consistency

- Follow existing naming patterns.
- Maintain backward compatibility unless explicitly allowed.
- Prefer functional utilities over complex classes unless justified.

### Performance

- Avoid unnecessary copies of NumPy arrays.
- Prefer vectorized operations over Python loops in hot paths.
- Use OpenCV operations efficiently.

---

## 3. Implementing Features

- Provide a minimal, clean implementation.
- Include type hints and Google-style docstrings with usage examples.
- All new functionality must be covered with tests, including edge cases.
- Add or update documentation (docstrings + mkdocs entries if applicable).
- Ensure compatibility with core dependencies: NumPy, OpenCV, SciPy.

---

## 4. Fixing Bugs

1. Reproduce and understand the root cause.
2. Write a test that reproduces the bug (it should fail before the fix).
3. Apply a minimal, targeted fix.
4. Verify the test passes and no other components break.

---

## 5. Refactoring

- Preserve behavior and API stability.
- Improve readability or performance.
- Reduce duplication.
- Avoid large, sweeping refactors unless explicitly requested.

---

## 6. Before You Commit

Always run these before committing:

```bash
uv run pytest --cov=supervision
uv run pre-commit run --all-files
```

- All pre-commit hooks must pass (formatting, linting, type checking, spell check, etc.).
- All tests must pass before opening a PR. Note: some existing tests in the repo may
    already be failing — your changes must not introduce new failures.
- Fix any issues reported and re-run until clean.

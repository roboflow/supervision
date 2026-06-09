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

- **Heading depth in docs/docstrings**: `###` maximum. `####` and deeper render
    identically to bold in mkdocs — use `**bold**` instead.

- **Formatting and linting** are enforced by **pre-commit**.
    The hook chain typically includes: ruff-check, ruff-format, codespell, mdformat,
    prettier, pyproject-fmt, and standard pre-commit-hooks (trailing whitespace, YAML, TOML, etc.).

- **Type hints**: required on all new code. Type checking with mypy is encouraged but not
    currently enforced systematically by pre-commit; see [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)
    for the latest type-checking expectations.

- **Docstrings**: Google Python docstring style. Required for all new functions and classes.
    Every docstring should include a usage example. Prefer `>>>` doctest format when
    the example only uses `supervision`, NumPy, and stdlib (no optional extras, no
    external files or network). See §3a and CONTRIBUTING.md for syntax.

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

## 3a. Test Conventions

Full test guidelines are in [CONTRIBUTING.md](.github/CONTRIBUTING.md#tests). Key rules:

- **AAA structure**: one arrange, one act, one assertion group per test. No second act.
- **Class grouping**: group related tests into a class. Class name = unit under test.
    Method names describe the expected outcome only — not the mechanism.
- **Parametrize**: 3+ structurally identical tests → `@pytest.mark.parametrize`.
    Use `pytest.param(..., id="slug")` per case (not `ids=[...]` on the decorator).
- **Docstrings**: every test function/method needs at minimum a one-line docstring
    within the project line length (see `pyproject.toml`). Describe the scenario, not the implementation.
- **Doctests**: prefer `>>>` doctest when example uses only `supervision`, NumPy, and
    stdlib (no optional extras, no external files). Fenced ```` ```python ```` is fine
    when non-runnable (third-party model, video file, optional extra) or when the
    example's purpose is showing exception/error behaviour. See CONTRIBUTING.md
    §Doctests for syntax guide (continuation lines, ELLIPSIS, `+SKIP` rules).

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

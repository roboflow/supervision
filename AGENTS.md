# Agent Guidelines for `supervision`

These instructions define how AI agents (GitHub Copilot, Claude, etc.) should behave when assigned an issue, task, or multi-step problem in this repository.

Behave like a senior contributor: precise, efficient, aligned with the project's philosophy, and focused on maintainability and clarity.

When this file and [CONTRIBUTING.md](.github/CONTRIBUTING.md) conflict, **CONTRIBUTING.md wins**.

---

## 1. Before You Code

- Read the task/issue thoroughly before acting.
- Identify missing information; group related clarifications into one structured ask — avoid sequential drip questions.
- Outline a step-by-step plan before making changes.
- Check whether the feature or fix already exists under a different name.
- Confirm alignment with the repository's architecture (`src/supervision/`).

---

## 2. Repository Architecture

**Package root**: `src/supervision/` — all library code lives here. **Tests**: `tests/` — mirrors the `src/supervision/` directory structure. **Public API**: everything re-exported from `src/supervision/__init__.py`.

### Core modules

| Module                      | Purpose                                                                                                                                                                                                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `detection/core.py`         | `Detections` dataclass — the central data structure. Holds `xyxy`, `mask`, `confidence`, `class_id`, `tracker_id`, and an open `data` dict for extra fields. All model connectors (`from_ultralytics`, `from_inference`, `from_transformers`, etc.) are classmethods here. |
| `detection/compact_mask.py` | Compact mask representation (47.9 KB — large, active module).                                                                                                                                                                                                              |
| `detection/line_zone.py`    | `LineZone` and related crossing-counting logic (top-level, not under `tools/`).                                                                                                                                                                                            |
| `detection/vlm.py`          | Connectors for vision-language models (Florence-2, Gemini, Qwen, PaliGemma, etc.).                                                                                                                                                                                         |
| `detection/utils/`          | Pure NumPy helpers: `boxes.py`, `converters.py`, `iou_and_nms.py`, `masks.py`, `polygons.py`, `internal.py`, `vlms.py`.                                                                                                                                                    |
| `detection/tools/`          | Higher-level tools: `InferenceSlicer`, `PolygonZone`, `CSVSink`, `JSONSink`, `DetectionsSmoother`.                                                                                                                                                                         |
| `annotators/core.py`        | All annotator classes (`BoxAnnotator`, `MaskAnnotator`, `LabelAnnotator`, …). Each implements `.annotate(scene, detections)`.                                                                                                                                              |
| `key_points/`               | Keypoint data structures and annotators (`KeyPoints`, `EdgeAnnotator`, `VertexAnnotator`, etc.). **Use this path — see §Deprecated module aliases below.**                                                                                                                 |
| `tracker/`                  | ByteTrack implementation.                                                                                                                                                                                                                                                  |
| `dataset/core.py`           | `DetectionDataset` / `ClassificationDataset` — load, split, merge, save in YOLO / COCO / Pascal VOC formats.                                                                                                                                                               |
| `geometry/core.py`          | `Point`, `Rect`, `Vector`, `Position` — shared geometry primitives.                                                                                                                                                                                                        |
| `metrics/`                  | Detection metrics (mAP, confusion matrix). Requires `pandas` (`--extra metrics`).                                                                                                                                                                                          |
| `utils/internal.py`         | Deprecation utilities (`warn_deprecated`, `deprecated_parameter`) and other internal helpers.                                                                                                                                                                              |
| `validators.py`             | Field validation helpers used by `Detections`.                                                                                                                                                                                                                             |
| `config.py`                 | Global string constants — always import from here, never use string literals.                                                                                                                                                                                              |

### Key design patterns

- **`Detections` is the lingua franca**: every model connector, tracker, and annotator speaks `Detections`. Adding a new connector means writing a `@classmethod from_<framework>(cls, result) -> Detections`.
- **Annotators are composable**: each annotator receives `scene` (a BGR `np.ndarray`) and `detections`, draws in-place on a copy, and returns the result.
- **`data` dict extensibility**: arbitrary per-detection metadata is stored in `detections.data` as `np.ndarray` aligned with `xyxy`. Key strings are constants in `config.py`.
- **Vectorized throughout**: NumPy arrays, no Python loops in hot paths. Never introduce `for det in detections` patterns.

---

## 3. Repository Conventions

All work must follow the conventions of the `supervision` library (see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for full details).

### Branching & Commits

- Branch from `develop` using prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`.
- Use **conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`.
- PRs must target the `develop` branch.

### Code Style

- **Heading depth in docs/docstrings**: `###` maximum. `####` and deeper render identically to bold in mkdocs — use `**bold**` instead.

- **Formatting and linting** are enforced by **pre-commit**. Canonical hook list is in `.pre-commit-config.yaml`; it includes: `ruff-check`, `ruff-format`, `mypy`, `mdformat`, `prettier`, `pyproject-fmt`, `codespell`, and standard pre-commit-hooks.

- **Type hints**: required on all new code. mypy is enforced by pre-commit.

- **Docstrings**: Google Python docstring style. Required for all new functions and classes. Every docstring should include a usage example. Prefer `>>>` doctest format when the example only uses `supervision`, NumPy, and stdlib (no optional extras, no external files or network). See §Doctest rules below and CONTRIBUTING.md §Doctests for full syntax guide.

### API Consistency

- Follow existing naming patterns.
- Maintain backward compatibility unless explicitly allowed.
- Prefer stateless functions in `detection/utils/`; promote to a class only when state crosses ≥2 method calls or matches an existing class pattern (e.g. annotator, tool).

### Runtime Dependencies

Runtime deps: `numpy`, `opencv-python`, `pillow`, `pyyaml`, `requests`, `scipy`, `tqdm`, `pydeprecate`, `defusedxml`, `matplotlib`. Optional extras: `metrics` (adds `pandas`).

Lazy-import any heavy framework dep (torch, transformers, ultralytics) inside the classmethod or function that needs it — never at module top level.

### Performance

- Avoid unnecessary copies of NumPy arrays.
- Prefer vectorized operations over Python loops in hot paths.
- Use OpenCV operations efficiently.

---

## 4. Test Conventions

These rules apply to **all contributions** — features, bug fixes, and refactors.

Full test guidelines are in [CONTRIBUTING.md](.github/CONTRIBUTING.md#tests). Key rules:

- **AAA structure**: one arrange, one act, one assertion group per test. No second act.
- **Class grouping**: group related tests into a class. Class name = unit under test. Method names describe the expected outcome only — not the mechanism.
- **Parametrize**: 3+ structurally identical tests → `@pytest.mark.parametrize`. Use `pytest.param(..., id="slug")` per case (not `ids=[...]` on the decorator).
- **Docstrings**: every test function/method needs at minimum a one-line docstring within the project line length (see `pyproject.toml`). Describe the scenario, not the implementation.

### Doctest rules

Prefer `>>>` doctest when example uses only `supervision`, NumPy, and stdlib. Output **must be deterministic**:

- Use `# doctest: +ELLIPSIS` for floats that vary by platform.
- Seed any RNG before calling it.
- Never assert `dict` or `set` iteration order.
- Never touch network or filesystem outside `supervision/assets/`.

Fenced ```` ```python ```` is correct when the example uses a third-party model, video file, optional extra, or intentionally shows exception/error behaviour.

---

## 5. Implementing Features

- Provide a minimal, clean implementation.
- Include type hints and Google-style docstrings with usage examples.
- Cover all new functionality with tests, including edge cases (see §4).
- Add or update documentation (docstrings + mkdocs entries if applicable).
- Ensure compatibility with all runtime dependencies listed above.

### Extending `Detections`

When adding per-detection metadata:

1. Store it in `detections.data` as `np.ndarray` aligned with `xyxy` (same first dimension).
2. Define the key string as a constant in `supervision/config.py`; import and use the constant — never a string literal.
3. OBB coordinates live under `data[ORIENTED_BOX_COORDINATES]` with shape `(N, 4, 2)`.

```python
# config.py
ORIENTED_BOX_COORDINATES = "xyxyxyxy"
CLASS_NAME_DATA_FIELD = "class_name"
```

### Adding a New Model Connector

Add a classmethod to `Detections` in `detection/core.py`:

```python
@classmethod
def from_myframework(cls, result) -> "Detections":
    # lazy-import heavy dep
    import myframework  # noqa: F401

    xyxy = ...  # np.ndarray shape (N, 4)
    confidence = ...
    class_id = ...
    data = {CLASS_NAME_DATA_FIELD: np.array([...])}
    return cls(xyxy=xyxy, confidence=confidence, class_id=class_id, data=data)
```

VLM connectors (Florence-2, Gemini, etc.) live in `detection/vlm.py` — add new VLMs there, not in `core.py`.

---

## 6. Deprecated Module Aliases

The `supervision.keypoint` module is deprecated since `0.27.0` and **will be removed in `0.30.0`**. Always import from `supervision.key_points` instead:

```python
# Wrong — deprecated, removed in 0.30.0
from supervision.keypoint import KeyPoints

# Correct
from supervision.key_points import KeyPoints
```

Both directories coexist on disk; `supervision.keypoint/__init__.py` emits a `SupervisionWarnings` on import via `warn_deprecated`. The public `supervision/__init__.py` re-exports only from `key_points`.

---

## 7. Deprecating APIs

Use the appropriate mechanism depending on context:

| What                      | Mechanism                                         | Location                    |
| ------------------------- | ------------------------------------------------- | --------------------------- |
| Module-level deprecation  | `supervision.utils.internal.warn_deprecated`      | module `__init__.py`        |
| Function/method parameter | `supervision.utils.internal.deprecated_parameter` | decorator on function       |
| Public function or class  | `@deprecated` from `pydeprecate` package          | decorator on function/class |

Always specify the version introduced and the version of removal, e.g.:

```python
warn_deprecated(
    "The 'foo' function is deprecated in `0.27.0` and will be removed in `0.30.0`. "
    "Use 'bar' instead."
)
```

---

## 8. Fixing Bugs

1. Reproduce and understand the root cause.
2. Write a test that reproduces the bug (it should fail before the fix).
3. Apply a minimal, targeted fix.
4. Verify the test passes and no other components break.

---

## 9. Refactoring

- Preserve behavior and API stability.
- Improve readability or performance.
- Reduce duplication.
- Avoid large, sweeping refactors unless explicitly requested.
- When changing or removing public API, apply the deprecation machinery from §7 above.

---

## 10. Before You Commit

Always run these before committing:

```bash
uv run pytest --cov=supervision
uv run pre-commit run --all-files
```

- All pre-commit hooks must pass (formatting, linting, type checking, spell check, etc.).
- All tests must pass. Note: some existing tests may already be failing — **your changes must not introduce new failures**. To verify, capture a baseline before making changes:

```bash
git stash && uv run pytest -q 2>&1 | tee /tmp/baseline.txt && git stash pop
# after changes:
uv run pytest -q 2>&1 | tee /tmp/after.txt
diff /tmp/baseline.txt /tmp/after.txt
```

Any test that passes in the baseline and fails after your changes is a blocker.

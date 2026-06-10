# Agent Guidelines for `supervision`

Behave like a senior contributor: precise, efficient, maintainable. When this file and [CONTRIBUTING.md](.github/CONTRIBUTING.md) conflict, **CONTRIBUTING.md wins**.

---

## 1. Before You Code

- Read the task thoroughly; group clarifications into one ask.
- Outline a plan before making changes.
- Check whether the feature already exists under a different name.
- Confirm alignment with `src/supervision/` architecture.

---

## 2. Repository Architecture

**Package root**: `src/supervision/` — all library code. **Tests**: `tests/` — mirrors `src/supervision/`. **Public API**: `src/supervision/__init__.py`.

### Core modules

```
src/supervision/
├── detection/
│   ├── core.py          — Detections dataclass; all model connectors as classmethods
│   ├── compact_mask.py  — compact mask representation
│   ├── vlm.py           — VLM connectors (Florence-2, Gemini, Qwen, PaliGemma)
│   ├── utils/           — pure NumPy helpers: boxes, converters, iou_and_nms, masks, polygons
│   └── tools/           — InferenceSlicer, PolygonZone, LineZone, CSVSink, JSONSink, DetectionsSmoother
├── annotators/core.py   — BoxAnnotator, MaskAnnotator, LabelAnnotator, … each: .annotate(scene, detections)
├── key_points/          — KeyPoints, EdgeAnnotator, VertexAnnotator (use this, NOT keypoint/ — see §6)
├── tracker/             — ByteTrack
├── dataset/core.py      — DetectionDataset / ClassificationDataset (YOLO / COCO / Pascal VOC)
├── geometry/core.py     — Point, Rect, Vector, Position
├── metrics/             — mAP, confusion matrix (requires --extra metrics)
├── utils/internal.py    — warn_deprecated, deprecated_parameter, internal helpers
└── config.py            — string constants; always import from here, never use literals
```

### Key design patterns

- **`Detections` is the lingua franca** — every connector, tracker, and annotator speaks `Detections`. New connector = `@classmethod from_<framework>(cls, result) -> Detections`.
- **Annotators are composable** — receive `scene` (BGR `np.ndarray`) + `detections`, return annotated copy.
- **`data` dict extensibility** — per-detection metadata in `detections.data` as `np.ndarray` aligned with `xyxy`. Keys are constants from `config.py`.
- **Vectorized throughout** — NumPy arrays, no Python loops in hot paths. Never write `for det in detections`.

---

## 3. Conventions

### Branching & Commits

- Branch from `develop`: `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`.
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`.
- PRs target `develop`.

### Code Style

- Doc headings: `###` max. Use `**bold**` instead of `####`.
- Formatting/linting via **pre-commit** (`ruff`, `mypy`, `mdformat`, `prettier`, `codespell`).
- Type hints required on all new code.
- Docstrings: Google style, usage example required. Use `>>>` doctest when example uses only `supervision`, NumPy, and stdlib. Use fenced ```` ```python ```` for third-party models, files, or intentional exceptions.

### API & Dependencies

- Follow existing naming patterns; maintain backward compatibility.
- Prefer stateless functions in `detection/utils/`; use a class only when state spans ≥2 calls.
- Runtime deps: `numpy`, `opencv-python`, `pillow`, `pyyaml`, `requests`, `scipy`, `tqdm`, `pydeprecate`, `defusedxml`, `matplotlib`.
- Lazy-import heavy deps (`torch`, `transformers`, `ultralytics`) inside the function that needs them — never at module top level.

### Performance

- No unnecessary NumPy copies.
- Vectorize hot paths; use OpenCV efficiently.

---

## 4. Tests

- **AAA structure**: one arrange, one act, one assertion group. No second act.
- Group tests in a class named after the unit under test. Method names describe the outcome.
- 3+ identical tests → `@pytest.mark.parametrize` with `pytest.param(..., id="slug")`.
- Every test needs a one-line docstring describing the scenario.

**Doctest rules**: output must be deterministic. Use `# doctest: +ELLIPSIS` for platform floats; seed any RNG; never assert `dict`/`set` order; no network or filesystem outside `supervision/assets/`.

---

## 5. Implementing Features

- Minimal, clean implementation with type hints and Google docstrings.
- Tests covering new functionality and edge cases.
- Update docstrings and mkdocs entries as needed.

**Extending `Detections`**: store metadata in `detections.data` as `np.ndarray` aligned with `xyxy`; define the key constant in `config.py`.

```python
# config.py
ORIENTED_BOX_COORDINATES = "xyxyxyxy"
CLASS_NAME_DATA_FIELD = "class_name"
```

**New model connector** (`detection/core.py`):

```python
@classmethod
def from_myframework(cls, result) -> "Detections":
    import myframework  # noqa: F401

    xyxy = ...  # (N, 4)
    return cls(
        xyxy=xyxy,
        confidence=...,
        class_id=...,
        data={CLASS_NAME_DATA_FIELD: np.array([...])},
    )
```

VLM connectors go in `detection/vlm.py`, not `core.py`.

---

## 6. Deprecated Module Aliases

`supervision.keypoint` deprecated since `0.27.0`, removed in `0.30.0`. Use `supervision.key_points`:

```python
from supervision.key_points import KeyPoints  # correct
```

---

## 7. Deprecating APIs

- Module-level: `supervision.utils.internal.warn_deprecated` in `__init__.py`
- Function/method parameter: `supervision.utils.internal.deprecated_parameter` decorator
- Public function or class: `@deprecated` from `pydeprecate`

Always name the version introduced and the removal version:

```python
warn_deprecated("'foo' deprecated in `0.27.0`, removed in `0.30.0`. Use 'bar'.")
```

---

## 8. Bugs & Refactoring

**Bugs**: reproduce → write failing test → minimal fix → verify no regressions.

**Refactoring**: preserve behavior and API; reduce duplication; avoid sweeping changes unless requested; apply §7 deprecation when removing public API.

---

## 9. Before You Commit

```bash
uv run pytest --cov=supervision
uv run pre-commit run --all-files
```

Capture a baseline to avoid introducing new failures:

```bash
git stash && uv run pytest -q 2>&1 | tee /tmp/baseline.txt && git stash pop
uv run pytest -q 2>&1 | tee /tmp/after.txt
diff /tmp/baseline.txt /tmp/after.txt
```

Any test passing in baseline but failing after = blocker.

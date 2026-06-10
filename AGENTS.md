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

```
src/supervision/
├── detection/
│   ├── core.py          — Detections dataclass; all model connectors as classmethods
│   ├── compact_mask.py  — compact mask representation
│   ├── vlm.py           — VLM connectors (Florence-2, Gemini, Qwen, PaliGemma)
│   ├── utils/           — pure NumPy helpers: boxes, converters, iou_and_nms, masks, polygons
│   ├── line_zone.py     — LineZone
│   └── tools/           — InferenceSlicer, PolygonZone, CSVSink, JSONSink, DetectionsSmoother
├── annotators/core.py   — BoxAnnotator, MaskAnnotator, LabelAnnotator, … each: .annotate(scene, detections)
├── key_points/          — KeyPoints, EdgeAnnotator, VertexAnnotator (use this, NOT keypoint/ — see §4)
├── tracker/             — DEPRECATED
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
- **Lazy-import heavy deps** — `torch`, `transformers`, `ultralytics` must be imported inside the function that needs them, never at module top level.

---

## 3. Agent-Critical Rules

These supplement [CONTRIBUTING.md](.github/CONTRIBUTING.md) — covering gaps or agent-specific failure modes.

**Doc headings**: `###` max in docstrings and docs. `####` renders identically to bold in mkdocs — use `**bold**` instead.

**Type hints**: required on all new code. mypy is enforced by pre-commit (`.pre-commit-config.yaml`).

**Doctest determinism** — output must be reproducible across platforms:

- Use `# doctest: +ELLIPSIS` for floats that vary by platform.
- Seed any RNG before calling it.
- Never assert `dict` or `set` iteration order.
- No network or filesystem access outside `supervision/assets/`.

**⚠ Test structure** — agents frequently fail here; read [CONTRIBUTING.md §Tests](.github/CONTRIBUTING.md#-tests) carefully: AAA structure, class grouping, parametrize with `pytest.param(..., id="slug")`, one-line docstring per test.

For branching, commit, code style, and API design conventions see [CONTRIBUTING.md](.github/CONTRIBUTING.md).

---

## 4. Deprecated Module Aliases

`supervision.keypoint` deprecated since `0.27.0`, removed in `0.30.0`. Always import from `supervision.key_points`, not `supervision.keypoint`.

---

## 5. Deprecating APIs

- Module-level: `supervision.utils.internal.warn_deprecated` in the deprecated module's own `__init__.py`
- Parameter renamed (old→new): `supervision.utils.internal.deprecated_parameter` decorator
- Public function, method, or class: `@deprecated` from `pydeprecate`

Always name the version introduced and the removal version:

```python
warn_deprecated("'foo' deprecated in `0.27.0`, removed in `0.30.0`. Use 'bar'.")
```

---

## 6. Implementing Features

- Minimal implementation; type hints and Google docstrings with usage examples.
- Tests covering new functionality and edge cases (see [CONTRIBUTING.md §Tests](.github/CONTRIBUTING.md#-tests)).
- Update docstrings and mkdocs entries as needed.

**Extending `Detections`**: store metadata in `detections.data` as `np.ndarray` aligned with `xyxy`; define the key as a constant in `config.py` (e.g. `CLASS_NAME_DATA_FIELD`, `ORIENTED_BOX_COORDINATES`).

**New model connector** (`detection/core.py`):

```python
@classmethod
def from_myframework(cls, result) -> "Detections":
    import myframework  # noqa: F401 — lazy import

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

## 7. Bugs & Refactoring

**Bugs**: reproduce → write failing test → minimal fix → verify no regressions.

**Refactoring**: preserve behavior and API; reduce duplication; avoid sweeping changes unless requested; apply §5 deprecation when removing public API.

---

## 8. Before You Commit

```bash
uv run pytest --cov=supervision
uv run pre-commit run --all-files
```

Capture a baseline before changes to avoid introducing new failures:

```bash
STASH_BEFORE=$(git rev-parse refs/stash 2>/dev/null)
git stash push --include-untracked
uv run pytest -q 2>&1 | tee /tmp/baseline.txt
[ "$(git rev-parse refs/stash 2>/dev/null)" != "$STASH_BEFORE" ] && git stash pop
uv run pytest -q 2>&1 | tee /tmp/after.txt
diff /tmp/baseline.txt /tmp/after.txt
```

Any test passing in baseline but failing after = blocker.

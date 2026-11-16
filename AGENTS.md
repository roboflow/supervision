# Agent Guidelines for `supervision`

Behave like a senior contributor: precise, efficient, maintainable. When this file and [CONTRIBUTING.md](.github/CONTRIBUTING.md) conflict, **CONTRIBUTING.md wins**.

______________________________________________________________________

## 1. Before You Code

- Read the task thoroughly; group clarifications into one ask.
- Outline a plan before making changes.
- Check whether the feature already exists under a different name.
- Confirm alignment with `src/supervision/` architecture.
- For pull request reviews, follow [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md).

______________________________________________________________________

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

______________________________________________________________________

## 3. Agent-Critical Rules

These supplement [CONTRIBUTING.md](.github/CONTRIBUTING.md) — covering gaps or agent-specific failure modes.

**Doc headings**: `###` max in docstrings and docs. `####` renders identically to bold in mkdocs — use `**bold**` instead.

**Docstring API references**: reference local classes, functions, and methods with Sphinx roles (`:class:`, `:func:`, or `:meth:`) and fully qualified targets rooted at `supervision.` rather than plain code literals or relative names. Use `~` to display a short name, for example, \`\`:class:\`~supervision.detection.core.Detections\`\`\`.

**Type hints**: required on all new code. mypy is enforced by pre-commit (`.pre-commit-config.yaml`).

**Function docstrings**: every new or modified function, including private helpers and tests, must have a succinct docstring explaining its purpose. Put function-level why/what/how context inside the function docstring, not in a comment before the function. Public APIs still require the full Google-style structure described below.

**Readable argument lists**: do not put multi-branch conditional expressions inside function or constructor arguments. If an argument needs more than a simple `a if condition else b`, assign it to a named local variable before the call.

**Inline comments**: write code so the intent is clear from names, small helpers, and straightforward control flow. For non-trivial logic inside a function that still needs context, add concise inline comments explaining why the code exists, what invariant it protects, and how the tricky part works. Do not put comments before functions; use the function docstring instead. Do not comment obvious assignments, mechanical plumbing, lint-only changes, typing-only changes, or pure docs edits.

**Doctest determinism** — output must be reproducible across platforms:

- Use `# doctest: +ELLIPSIS` for floats that vary by platform.
- Seed any RNG before calling it.
- Never assert `dict` or `set` iteration order.
- No network or filesystem access outside `supervision/assets/`.

**⚠ Test structure** — agents frequently fail here; read [CONTRIBUTING.md §Tests](.github/CONTRIBUTING.md#-tests) carefully: AAA structure, class grouping, bare simple parametrized values, `pytest.param(..., id="semantic-slug")` for compound or unclear cases (never parallel `ids`), one-line docstring per test.

**Reference parity tests** — follow [CONTRIBUTING.md §Reference parity tests](.github/CONTRIBUTING.md#reference-parity-tests): evaluate the live reference package on the same inputs, and keep it in development dependencies only.

For branching, commit, code style, and API design conventions see [CONTRIBUTING.md](.github/CONTRIBUTING.md).

______________________________________________________________________

## 4. Deprecated Module Aliases

`supervision.keypoint` deprecated since `0.27.0`, removed in `0.31.0`. Always import from `supervision.key_points`, not `supervision.keypoint`.

______________________________________________________________________

## 5. Deprecating APIs

**Minimum window**: deprecated APIs must remain for at least **3 minor releases** before removal. Example: deprecated in `0.29.0` → removed in `0.32.0`.

- Module-level: `supervision.utils.internal.warn_deprecated` in the deprecated module's own `__init__.py`
- Parameter renamed (old→new): `supervision.utils.internal.deprecated_parameter` decorator
- Public function, method, or class: `@deprecated` from `pydeprecate`

Always name the version introduced and the removal version:

```python
warn_deprecated("'foo' deprecated in `0.29.0`, removed in `0.32.0`. Use 'bar'.")
```

______________________________________________________________________

## 6. Implementing Features

- Minimal implementation; type hints and Google docstrings with usage examples.
- Tests covering new functionality and edge cases (see [CONTRIBUTING.md §Tests](.github/CONTRIBUTING.md#-tests)).
- Update docstrings and mkdocs entries as needed.
- Update [docs/changelog.md](docs/changelog.md) for every functional change or bug fix, including user-visible behavior changes. Skip changelog entries for lint-only, type-only, formatting-only, and pure documentation-only changes.
- If work has open PR, changelog entry must link that PR (`[#N](https://github.com/roboflow/supervision/pull/N)`), not source issue. PR link only, even if issue exists.

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

______________________________________________________________________

## 7. Bugs & Refactoring

**Bugs**: reproduce → write failing test → minimal fix → verify no regressions.

**Refactoring**: preserve behavior and API; reduce duplication; avoid sweeping changes unless requested; apply §5 deprecation when removing public API.

______________________________________________________________________

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

## GitHub Actions

- Keep public CI on standard GitHub-hosted runners.
- Cancel superseded PR runs with a workflow-specific concurrency group. Preserve every push test run for regression tracing. Serialize documentation publishing without cancelling active pushes.
- Apply path filters only to PR events. Filter PR link checks to Markdown and their configuration. Filter PR GitHub configuration tests to their scripts, fixtures, documentation, and dependency inputs. Keep required checks triggered for every PR so path filters cannot block merging.
- Keep the full Python version matrix on Linux, and the oldest and newest matrix versions on macOS and Windows. This reduces repeated OS and Python combinations while retaining every Python minor and platform. Preserve OpenCV backend and isolated wheel coverage on every OS.
- Set a timeout on every runner job, including reusable workflows and result guardians.
- Cache dependencies in test workflows. Disable caching when building or publishing releases so publishing jobs cannot restore caches from test runs. Retain distribution artifacts for three days.
- Pin actions to full commit SHAs with release-version comments. Keep the GitHub Actions Dependabot ecosystem enabled.
- Default workflow permissions to read-only and elevate individual jobs only when needed. Disable checkout credential persistence unless that checkout must push.
- Pass event values through environment variables and quote shell expansions. Never interpolate branch names, dispatch inputs, or release metadata into shell code.
- Keep the conflict labeler API-only. A `pull_request_target` job must never check out or execute PR code.

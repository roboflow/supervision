# Development Guide

Reference for contributors who work on the code, tests and documentation of Supervision. If this is your first contribution, start with [Your First Contribution](first_contribution.md).

## Set up your environment

1. Fork and clone the repository, and add the `upstream` remote, as described in [Your First Contribution](first_contribution.md).

2. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

3. Install the project with the development and documentation dependencies:

    ```bash
    uv sync --group dev --group docs --extra metrics
    ```

4. Verify the setup:

    ```bash
    uv run pytest
    ```

## API design principles

Supervision APIs should remain generic, composable, and predictable across model families. Before adding a new integration, annotator option, or data conversion method, check the existing `sv.Detections`, `sv.KeyPoints`, and annotator patterns and follow these principles:

1. **Model integrations normalize raw external outputs into existing Supervision containers.** Use `sv.Detections` for detection, segmentation, and other instance-level predictions that include boxes, masks, class ids, confidence scores, or extra per-instance fields. Use `sv.KeyPoints` for keypoint and pose predictions. Store an available per-instance confidence vector in `KeyPoints.detection_confidence`; leave it `None` when the source does not provide one. `sv.Detections` has no keypoint field; `KeyPoints.as_detections()` derives boxes from the key points when a `Detections` object is needed, for example for tracking.
2. **Do not add a `from_<model>` method when the model already returns a Supervision object.** `from_*` methods are for converting raw outputs from external packages such as Ultralytics, Transformers, Inference, or MediaPipe. If a model's `predict()` method already returns `sv.Detections`, keep that result type and store additional structured payloads in `detections.data` or `detections.metadata` using documented keys.
3. **Annotators render data; filtering and visibility are container state.** Filtering by confidence, class id, tracker id, geometry, or custom data should happen before annotation through the container slicing APIs. When detection confidence is available, use `detections[detections.confidence > 0.7]` or `key_points[key_points.detection_confidence > 0.5]` to select instances. Per-point presentation state, such as a `KeyPoints.visible` mask, may live on the container and be honored consistently by annotators.
4. **Annotator constructor arguments should describe visual presentation, not model-quality gates.** Use constructor arguments for color, thickness, opacity, text, position, style, and generic visualization parameters such as sigma levels. Annotators may skip invalid geometry defensively, including missing points, zero-area boxes, non-finite coordinates, or points marked invisible on the container. They should not introduce confidence thresholds or model-specific quality gates as rendering options.

## Code style and quality

### Pre-commit

The project uses [pre-commit](https://pre-commit.com/) to keep code quality consistent, and the same checks run in CI on every pull request. Run them before you commit:

```bash
uv run pre-commit run --all-files
```

If a hook reports an issue, fix it and re-run until everything passes. To run the checks automatically on every commit, install the git hook once:

```bash
uv run pre-commit install
```

### Docstrings

All new functions and classes need docstrings in the [Google Python docstring style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods), with a usage example. See [Doctests](#doctests) for how to write examples that the test suite verifies.

When a docstring refers to a local class, function, or method, use the Sphinx cross-reference role (`:class:`, `:func:`, or `:meth:`) with a fully qualified target rooted at `supervision.` instead of a plain code literal or relative name. Prefix the target with `~` to display only its short name, for example, \`\`:class:\`~supervision.detection.core.Detections\`\`\`.

### Type checking

Type hints are required on all new code. mypy runs in the pre-commit hooks, so your pull request fails CI if mypy reports errors.

### Readability

Avoid multi-branch conditional expressions inside function or constructor arguments. If an argument needs more than a simple `a if condition else b`, assign it to a named local variable before the call.

### Performance

- Avoid unnecessary copies of NumPy arrays.
- Prefer vectorized operations over Python loops in hot paths.
- Lazy-import heavy framework dependencies (`torch`, `transformers`, `ultralytics`) inside the function that needs them, never at module top level.

### Deprecation policy

Deprecated APIs must stay for at least **3 minor releases** before removal. For example, an API deprecated in `0.29.0` can be removed in `0.32.0`. Always state both the deprecation version and the planned removal version.

- **Module-level alias**: `supervision.utils.internal.warn_deprecated` in the deprecated module's `__init__.py`.
- **Renamed parameter**: the `supervision.utils.internal.deprecated_parameter` decorator.
- **Public function, method, or class**: `@deprecated` from `pydeprecate`.

## Documentation

The documentation lives in the `docs` folder and is built with [mkdocs](https://www.mkdocs.org/). To preview it locally:

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

- New public functions and classes need an entry in the docs so the API reference is generated, and a reference in the `mkdocs.yml` navigation where applicable.
- Add an entry to [`docs/changelog.md`](https://github.com/roboflow/supervision/blob/develop/docs/changelog.md) for every functional change or bug fix. Skip it for lint-only, type-only, formatting-only and documentation-only changes.

## Cookbooks

We are always looking for new examples that help others solve a real problem. To add a cookbook:

- Create a new notebook in the [`docs/notebooks`](https://github.com/roboflow/supervision/tree/develop/docs/notebooks) folder.
- Add a link to it in [`docs/theme/cookbooks.html`](https://github.com/roboflow/supervision/blob/develop/docs/theme/cookbooks.html), with the notebook path, a title, labels, author and supervision version.
- Use the [Count Objects Crossing the Line](https://supervision.roboflow.com/develop/notebooks/count-objects-crossing-the-line/) cookbook as a template, including the "Open in Colab" button at the top.
- Pin the `supervision` version the notebook uses.
- Keep the notebook self-contained: include the download and installation commands for any external data or libraries.
- Annotate the code with comments, linking to the documentation of each tool you use.

## Tests

[`pytest`](https://docs.pytest.org/en/7.1.x/) runs the tests:

```bash
uv run pytest
uv run pytest --cov=supervision  # with coverage
```

### Test structure

Follow **Arrange-Act-Assert (AAA)**: one setup block, one action, one assertion group per test. Never put two independent actions in the same test.

**Class grouping:** Group related tests into a class. The class name carries the unit under test; method names describe the expected outcome only, not the mechanism.

```python
class TestDetectionsWithNms:
    def test_keeps_highest_confidence_detection(self): ...
    def test_suppresses_lower_score_when_overlap_exceeds_threshold(self): ...
    def test_raises_when_confidence_missing(self): ...
```

**Parametrize aggressively:** Three or more structurally identical tests should become a single `@pytest.mark.parametrize` case. Use bare strings, numbers, booleans, and `None` values. Use `pytest.param(..., id="semantic-slug")` when a case passes a function, object, or compound setup; needs a per-case mark; or would otherwise produce an unclear or empty ID. Never use a parallel `ids=[...]` list: co-locate an explicit ID with its arguments so it survives reordering. Move a trailing comment that only names a case into that ID; retain comments only when they explain a non-obvious invariant.

```python
@pytest.mark.parametrize(
    ("overlap_metric", "expected_keep"),
    [
        pytest.param(OverlapMetric.IOU, [True, True], id="iou-keeps-both"),
        pytest.param(OverlapMetric.IOS, [True, False], id="ios-suppresses-small"),
    ],
)
def test_overlap_metric_determines_suppression(
    overlap_metric: OverlapMetric, expected_keep: list[bool]
) -> None:
    """Small box inside large: IOU keeps both; IOS suppresses small."""
    ...
```

**Docstrings:** Every test function or method needs at least a one-line docstring, within the project line length configured in `pyproject.toml`. Describe the scenario, not the implementation.

### Reference parity tests

When tests claim parity with or compare results to a reference implementation or golden standard, compute the reference results using the live package on the same inputs rather than hardcoded expected values. Keep reference packages in the development dependency group; do not add them to runtime dependencies or optional extras.

!!! note "Exception"

    A reference package that is also a documented, opt-in runtime backend may additionally appear in an optional extra. Currently this covers only `pycocotools` in the `coco` extra, which backs `mask_rasterizer="pycocotools"`; test-only reference packages stay in the development group. Revisit this exception when the default rasterizer changes (no earlier than `0.35.0`), after deciding whether to port the algorithm natively or make `pycocotools` a core dependency.

### Doctests

When an example uses only `supervision`, NumPy, and the standard library, with no optional extras, external files, network or devices, prefer `>>>` doctest format so the test suite verifies it automatically. Fenced ```` ```python ```` blocks are appropriate when the example cannot reasonably be executed (for example loading a third-party model or reading a video file) or when it demonstrates error behavior rather than return values.

Doctests run as part of the test suite via `--doctest-modules` in `pyproject.toml`. The `ELLIPSIS` and `NORMALIZE_WHITESPACE` flags are enabled globally, so `...` matches any output fragment and minor whitespace differences are ignored.

```bash
uv run pytest --doctest-modules src/
```

Use the `Example:` section of a Google-style docstring. Prefix each input line with `>>>` and each continuation line with `...`, and place the expected output immediately after the last input line:

```python
def clip_boxes(xyxy: np.ndarray, resolution_wh: tuple) -> np.ndarray:
    """Clip bounding boxes to frame boundaries.

    Args:
        xyxy: Box coordinates as (N, 4) float array.
        resolution_wh: Frame size as (width, height).

    Returns:
        Clipped boxes as (N, 4) float array.

    Example:
        >>> import numpy as np
        >>> import supervision as sv
        >>> boxes = np.array([[-10, -5, 120, 80]], dtype=np.float32)
        >>> sv.clip_boxes(boxes, resolution_wh=(100, 60))
        array([[ 0.,  0., 100.,  60.]], dtype=float32)
    """
```

Rules of thumb:

- **Print output**: write the printed string as expected output, without quotes.
- **`None` return**: no output line needed; suppress it with an assignment or `_ =`.
- **Large or variable arrays**: use `ELLIPSIS`, for example `array([...])`.
- **`# doctest: +SKIP`**: a last resort for genuinely non-runnable lines, such as a GPU-only call inside an otherwise runnable example. Prefer splitting the example into two blocks.
- **Fenced blocks** remain appropriate for examples that import optional extras (`supervision[metrics]`, `torch`, `ultralytics`), read files or capture video, or are intentionally incomplete pseudocode.

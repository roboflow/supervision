"""Integration checks for the OpenCV compatibility boundary."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from supervision import _cv2
from supervision._cv2._image import _add_weighted, _flip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "supervision"
TEST_ROOT = PROJECT_ROOT / "tests"


def _direct_cv2_imports(root: Path, excluded: set[Path] | None = None) -> list[str]:
    """Return direct cv2 import locations under a source or test tree."""
    excluded = excluded or set()
    imports: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            is_direct_import = isinstance(node, ast.Import) and any(
                alias.name == "cv2" for alias in node.names
            )
            is_direct_from_import = (
                isinstance(node, ast.ImportFrom) and node.module == "cv2"
            )
            if is_direct_import or is_direct_from_import:
                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                imports.append(f"{relative_path}:{node.lineno}")
    return imports


def _blocked_cv2_environment(tmp_path: Path) -> dict[str, str]:
    """Create a subprocess environment that rejects every cv2 import."""
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        (
            "import sys\n"
            "\n"
            "class BlockCv2:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            '        if fullname == "cv2":\n'
            '            raise ModuleNotFoundError("blocked for integration test")\n'
            "        return None\n"
            "\n"
            "\n"
            "sys.meta_path.insert(0, BlockCv2())\n"
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    python_path = [str(tmp_path), str(PROJECT_ROOT / "src")]
    if existing_path := environment.get("PYTHONPATH"):
        python_path.append(existing_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def test_production_imports_cv2_only_through_facade() -> None:
    """Keep native OpenCV imports inside the private facade module."""
    imports = _direct_cv2_imports(SOURCE_ROOT)

    assert all(
        location.startswith("src/supervision/_cv2/__init__.py:") for location in imports
    )


def test_ordinary_tests_use_facade_instead_of_native_cv2() -> None:
    """Keep ordinary fixtures and regression tests runnable without OpenCV."""
    reference_root = TEST_ROOT / "cv2"
    imports = _direct_cv2_imports(TEST_ROOT, excluded=set(reference_root.rglob("*.py")))

    assert imports == []


def test_fallback_add_weighted_accepts_opencv_keyword_names() -> None:
    """Accept OpenCV's public `src1` and `src2` parameter names."""
    source = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    other = np.full_like(source, 50)

    actual = _add_weighted(src1=source, alpha=0.5, src2=other, beta=0.5, gamma=10)
    expected = _add_weighted(source, 0.5, other, 0.5, 10)

    np.testing.assert_array_equal(actual, expected)


def test_fallback_flip_accepts_opencv_keyword_names() -> None:
    """Accept OpenCV's public `src` and `flipCode` parameter names."""
    source = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    actual = _flip(src=source, flipCode=-1)

    np.testing.assert_array_equal(actual, np.array([[4, 3], [2, 1]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("flip_code", "expected"),
    [
        pytest.param(
            2, np.array([[2, 1], [4, 3]], dtype=np.uint8), id="positive-horizontal"
        ),
        pytest.param(
            -2, np.array([[4, 3], [2, 1]], dtype=np.uint8), id="negative-both"
        ),
    ],
)
def test_fallback_flip_follows_sign_of_flip_code(
    flip_code: int, expected: np.ndarray
) -> None:
    """Flip by the sign of any positive or negative code, as OpenCV does."""
    source = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    actual = _flip(source, flip_code)

    np.testing.assert_array_equal(actual, expected)


def test_facade_maps_speed_example_source_onto_target() -> None:
    """Run the speed example's ViewTransformer calls through the active backend."""
    source = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
    target = np.array([[0, 0], [24, 0], [24, 249], [0, 249]])
    matrix = _cv2.getPerspectiveTransform(
        source.astype(np.float32), target.astype(np.float32)
    )
    points = source.reshape(-1, 1, 2).astype(np.float32)

    projected = _cv2.perspectiveTransform(points, matrix)

    # float32 output spacing near 249 is about 1.5e-5; 1e-3 leaves room for the
    # conditioning of a homography solved from 5000-pixel image coordinates.
    np.testing.assert_allclose(projected.reshape(-1, 2), target, rtol=0, atol=1e-3)


def test_ordinary_suite_passes_when_cv2_is_blocked(tmp_path: Path) -> None:
    """Run all non-reference tests in a process where cv2 cannot be imported."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "--ignore=tests/cv2",
            "-q",
            "--disable-warnings",
        ],
        cwd=PROJECT_ROOT,
        env=_blocked_cv2_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr

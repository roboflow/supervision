"""Integration checks for the OpenCV compatibility boundary."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

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
                relative_path = path.relative_to(PROJECT_ROOT)
                imports.append(f"{relative_path}:{node.lineno}")
    return imports


def _blocked_cv2_environment(tmp_path: Path) -> dict[str, str]:
    """Create a subprocess environment that rejects every cv2 import."""
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        """import sys


class BlockCv2:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2":
            raise ModuleNotFoundError("blocked for integration test")
        return None


sys.meta_path.insert(0, BlockCv2())
""",
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

    assert imports
    assert all(
        location.startswith("src/supervision/_cv2/__init__.py:") for location in imports
    )


def test_ordinary_tests_use_facade_instead_of_native_cv2() -> None:
    """Keep ordinary fixtures and regression tests runnable without OpenCV."""
    reference_root = TEST_ROOT / "cv2"
    imports = _direct_cv2_imports(TEST_ROOT, excluded=set(reference_root.rglob("*.py")))

    assert imports == []


def test_ordinary_suite_passes_when_cv2_is_blocked(tmp_path: Path) -> None:
    """Run all non-reference tests in a process where cv2 cannot be imported."""
    completed = subprocess.run(  # noqa: S603
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

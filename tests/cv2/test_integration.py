"""Integration checks for the OpenCV compatibility boundary."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from supervision import _cv2
from supervision._cv2._image import _add_weighted, _flip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "supervision"
TEST_ROOT = PROJECT_ROOT / "tests"

#: Leading bytes the fake OpenCV decoder accepts as an image signature.
_FAKE_IMAGE_SIGNATURE = b"\x89FAKEIMG"
#: Size of the non-image file whose contents must stay out of memory.
_LARGE_FILE_BYTES = 32 * 1024 * 1024
#: Allocation ceiling for rejecting that file after reading only its header.
_HEADER_ONLY_PEAK_BYTES = 1024 * 1024


def _fake_cv2_attribute(name: str) -> str:
    """Stand in for every OpenCV name the facade imports but these tests never call."""
    if name.startswith("__"):
        raise AttributeError(name)
    return name


def _fake_imdecode(buffer: np.ndarray, flags: int) -> tuple[bytes, int] | None:
    """Decode like OpenCV: check the signature, then read everything for an image."""
    if bytes(buffer[: len(_FAKE_IMAGE_SIGNATURE)]) != _FAKE_IMAGE_SIGNATURE:
        return None
    return bytes(buffer), flags


def _peak_traced_bytes(function: Callable[[], object]) -> tuple[object, int]:
    """Run a call and return its result with the peak memory it allocated."""
    was_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return result, peak - baseline


@pytest.fixture
def windows_opencv_facade(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load a private facade copy backed by a fake OpenCV on the Windows route."""
    fake_cv2 = ModuleType("cv2")
    vars(fake_cv2).update(__getattr__=_fake_cv2_attribute, imdecode=_fake_imdecode)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    spec = importlib.util.spec_from_file_location(
        "_cv2_facade_with_fake_opencv", _cv2.__file__
    )
    assert spec is not None
    assert spec.loader is not None
    facade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facade)
    monkeypatch.setattr(facade, "_is_non_ascii_windows_path", lambda filename: True)
    return facade


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
            if isinstance(node, (ast.Import, ast.ImportFrom)) and (
                is_direct_import or is_direct_from_import
            ):
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


class TestFallbackDatasetImageExport:
    """In-memory dataset images remain writable without OpenCV."""

    @pytest.mark.parametrize("channels", [1, 4])
    def test_yolo_export_writes_jpeg_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int
    ) -> None:
        """Export JPEGs for grayscale channel axes and BGRA dataset images."""
        from supervision import DetectionDataset, Detections
        from supervision._cv2._image import _imwrite

        image = np.full((8, 10, channels), 120, dtype=np.uint8)
        dataset = DetectionDataset(
            classes=["object"],
            images={"image.jpg": image},
            annotations={"image.jpg": Detections.empty()},
        )
        monkeypatch.setattr(_cv2, "imwrite", _imwrite)
        output = tmp_path / "images"

        dataset.as_yolo(images_directory_path=str(output))

        saved = _cv2.imread(str(output / "image.jpg"))
        assert saved is not None
        assert saved.shape == (8, 10, 3)
        np.testing.assert_array_equal(saved, np.full((8, 10, 3), 120, dtype=np.uint8))


class TestNonAsciiImagePaths:
    """Read and write images whose path has characters outside ASCII."""

    @pytest.mark.parametrize("folder_name", ["café", "猫", "фото"])
    def test_imwrite_then_imread_round_trips(
        self, tmp_path: Path, folder_name: str
    ) -> None:
        """Write an image into a non-ASCII folder and read the same pixels back."""
        image = np.random.default_rng(0).integers(0, 256, (4, 6, 3), dtype=np.uint8)
        image_path = tmp_path / folder_name / "image.png"
        image_path.parent.mkdir()

        written = _cv2.imwrite(str(image_path), image)

        assert written
        actual = _cv2.imread(str(image_path))
        assert actual is not None
        np.testing.assert_array_equal(actual, image)

    def test_imwrite_keeps_non_ascii_file_name(self, tmp_path: Path) -> None:
        """Name the written file as given instead of in another encoding."""
        image = np.zeros((4, 6, 3), dtype=np.uint8)

        written = _cv2.imwrite(str(tmp_path / "café.png"), image)

        assert written
        assert [path.name for path in tmp_path.iterdir()] == ["café.png"]

    def test_imread_returns_none_for_missing_non_ascii_path(
        self, tmp_path: Path
    ) -> None:
        """Report a missing file the way `cv2.imread` does."""
        assert _cv2.imread(str(tmp_path / "café.png")) is None

    def test_byte_route_round_trips_on_every_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Take the Windows route on any platform so it stays exercised in CI."""
        monkeypatch.setattr(
            _cv2, "_is_non_ascii_windows_path", lambda filename: True, raising=False
        )
        image = np.random.default_rng(0).integers(0, 256, (4, 6, 3), dtype=np.uint8)
        image_path = tmp_path / "image.png"

        written = _cv2.imwrite(str(image_path), image)

        assert written
        np.testing.assert_array_equal(_cv2.imread(str(image_path)), image)
        assert _cv2.imread(str(tmp_path / "missing.png")) is None
        assert not _cv2.imwrite(str(tmp_path / "missing" / "image.png"), image)


class TestWindowsNonAsciiImread:
    """The OpenCV `imread` wrapper reads non-ASCII Windows paths through NumPy."""

    def test_non_image_file_is_not_read_into_memory(
        self, tmp_path: Path, windows_opencv_facade: ModuleType
    ) -> None:
        """Reject a large non-image file after OpenCV inspects only its header."""
        path = tmp_path / "café.bin"
        with path.open("wb") as file:
            file.truncate(_LARGE_FILE_BYTES)

        result, peak = _peak_traced_bytes(
            lambda: windows_opencv_facade.imread(str(path), 1)
        )

        assert result is None
        assert peak < _HEADER_ONLY_PEAK_BYTES

    def test_decodes_file_bytes_with_given_flags(
        self, tmp_path: Path, windows_opencv_facade: ModuleType
    ) -> None:
        """Hand OpenCV's decoder the exact file contents and the read flags."""
        content = _FAKE_IMAGE_SIGNATURE + bytes(range(256))
        path = tmp_path / "café.png"
        path.write_bytes(content)

        decoded = windows_opencv_facade.imread(str(path), 0)

        assert decoded == (content, 0)

    def test_decodes_file_bytes_when_file_cannot_be_mapped(
        self,
        tmp_path: Path,
        windows_opencv_facade: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Read the whole file instead when its file system refuses a memory map."""
        content = _FAKE_IMAGE_SIGNATURE + bytes(range(256))
        path = tmp_path / "café.png"
        path.write_bytes(content)

        def refuse_mapping(*args: object, **kwargs: object) -> np.ndarray:
            """Fail the way a file system without memory-map support does."""
            raise OSError("mapping is not supported")

        monkeypatch.setattr(np, "memmap", refuse_mapping)

        decoded = windows_opencv_facade.imread(str(path), 0)

        assert decoded == (content, 0)

    @pytest.mark.parametrize(
        "content",
        [pytest.param(b"", id="empty-file"), pytest.param(None, id="missing-file")],
    )
    def test_returns_none_for_unreadable_file(
        self, tmp_path: Path, windows_opencv_facade: ModuleType, content: bytes | None
    ) -> None:
        """Report an empty or missing file as unreadable, as `cv2.imread` does."""
        path = tmp_path / "café.png"
        if content is not None:
            path.write_bytes(content)

        result = windows_opencv_facade.imread(str(path), 1)

        assert result is None

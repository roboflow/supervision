#!/usr/bin/env python3
"""Verify a built Supervision wheel works without OpenCV."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from email import message_from_bytes
from email.message import Message
from pathlib import Path

_MANIFEST_CHECKS = {
    "import-supervision",
    "no-cv2-module",
    "fallback-backend",
    "bgr-to-gray",
    "draw-rectangle",
    "required-pyav",
}


def _wheel_metadata(wheel: Path) -> Message:
    """Read the core metadata embedded in a wheel archive."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith("/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(
                f"expected one METADATA file in {wheel}, found {metadata_paths}"
            )
        return message_from_bytes(archive.read(metadata_paths[0]))


def _validate_metadata(wheel: Path) -> None:
    """Reject wheels that retain an OpenCV runtime requirement or extra."""
    metadata = _wheel_metadata(wheel)
    requirements = metadata.get_all("Requires-Dist", [])
    extras = metadata.get_all("Provides-Extra", [])
    if any("opencv" in requirement.lower() for requirement in requirements):
        raise ValueError(
            f"OpenCV runtime requirement remains in {wheel}: {requirements}"
        )
    if any("opencv" in extra.lower() for extra in extras):
        raise ValueError(f"OpenCV extra remains in {wheel}: {extras}")


def _validate_manifest(manifest: Path) -> None:
    """Keep the installed-wheel fallback smoke contract explicit and complete."""
    checks = {
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if checks != _MANIFEST_CHECKS:
        raise ValueError(
            f"unexpected fallback manifest {checks}; expected {_MANIFEST_CHECKS}"
        )


def _run_installed_wheel_probe(python: Path) -> None:
    """Exercise the installed fallback without allowing the source tree on sys.path."""
    source = """
import importlib.util
from importlib import metadata
from pathlib import Path

import av
import numpy as np
import supervision
from supervision import _cv2

package_path = Path(supervision.__file__).resolve()
if "site-packages" not in package_path.parts:
    raise AssertionError(
        f"supervision did not import from site-packages: {package_path}"
    )
if importlib.util.find_spec("cv2") is not None:
    raise AssertionError("cv2 is installed in the clean-wheel environment")
opencv_distributions = [
    distribution.metadata["Name"]
    for distribution in metadata.distributions()
    if "opencv" in distribution.metadata["Name"].lower()
]
if opencv_distributions:
    raise AssertionError(
        "OpenCV distributions remain in the clean-wheel environment: "
        f"{opencv_distributions}"
    )
if _cv2.BACKEND_NAME != "fallback":
    raise AssertionError(f"expected fallback backend, got {_cv2.BACKEND_NAME!r}")

image = np.array([[[0, 0, 255]]], dtype=np.uint8)
assert _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY).tolist() == [[76]]
canvas = np.zeros((3, 3, 3), dtype=np.uint8)
assert _cv2.rectangle(canvas, (0, 0), (2, 2), (1, 2, 3), -1) is canvas
assert canvas.tolist() == [[[1, 2, 3]] * 3] * 3
assert av.__version__
"""
    subprocess.run(  # noqa: S603 - the caller passes the clean CI interpreter explicitly.
        [str(python), "-c", source],
        check=True,
        cwd=Path.cwd().parent,
    )
    subprocess.run(  # noqa: S603 - the caller passes the clean CI interpreter explicitly.
        [str(python), "-m", "pip", "check"], check=True
    )


def main() -> None:
    """Validate one wheel against a previously prepared clean environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    _validate_metadata(args.wheel)
    _validate_manifest(args.manifest)
    _run_installed_wheel_probe(args.python)


if __name__ == "__main__":
    main()

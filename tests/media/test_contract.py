"""Validate the PR0 media contract and its deterministic reference fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONTRACT_PATH = FIXTURE_DIR / "media_contract.json"
IMAGE_FIXTURE_PATH = FIXTURE_DIR / "bgr_2x2.ppm"


def _load_contract() -> dict[str, Any]:
    """Load the versioned media contract fixture."""
    with CONTRACT_PATH.open(encoding="utf-8") as contract_file:
        return json.load(contract_file)


def _as_array(value: dict[str, Any]) -> np.ndarray:
    """Decode a JSON array fixture into its declared NumPy dtype."""
    return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))


CONTRACT = _load_contract()
REFERENCE_CASES = tuple(
    pytest.param(case, id=case["id"]) for case in CONTRACT["reference_cases"]
)


def test_reference_probe_exposes_all_current_media_symbols() -> None:
    """The canonical reference environment exposes every inventoried symbol."""
    missing = [
        symbol
        for symbol in CONTRACT["reference"]["required_symbols"]
        if not hasattr(cv2, symbol)
    ]

    assert not missing, f"OpenCV reference probe is missing symbols: {missing}"


def test_contract_has_unique_operations_and_complete_symbol_mapping() -> None:
    """Every inventoried cv2 symbol maps to one semantic contract operation."""
    operations = CONTRACT["operations"]
    names = [operation["name"] for operation in operations]
    mapped_symbols = {
        symbol for operation in operations for symbol in operation["cv2_symbols"]
    }

    assert len(names) == len(set(names))
    assert set(CONTRACT["reference"]["required_symbols"]) <= mapped_symbols
    assert all(
        operation["signature"].startswith(operation["name"]) for operation in operations
    )


def test_contract_records_valid_parity_tiers_and_tolerances() -> None:
    """Each parity claim has a machine-readable tier and tolerance policy."""
    valid_tiers = set(CONTRACT["parity_tiers"])

    for operation in CONTRACT["operations"]:
        assert operation["input_contract"]
        assert operation["output_contract"]
        assert operation["error_contract"]
        for parity in operation["parity"]:
            assert parity["tier"] in valid_tiers
            if parity["tier"] == "D":
                assert parity["notes"]
                if parity["max_abs"] is None or parity["mean_abs"] is None:
                    assert parity["max_abs"] is None
                    assert parity["mean_abs"] is None
                    continue
            assert parity["max_abs"] is not None
            assert parity["mean_abs"] is not None
            assert parity["max_abs"] >= 0
            assert parity["mean_abs"] >= 0
            assert parity["mean_abs"] <= parity["max_abs"]


@pytest.mark.parametrize("case", REFERENCE_CASES)
def test_reference_case_matches_canonical_opencv(case: dict[str, Any]) -> None:
    """Reference arrays remain stable under the selected OpenCV oracle."""
    operation = case["operation"]
    image = _as_array(case["input"])

    if operation == "convert_color":
        result = cv2.cvtColor(image, getattr(cv2, case["cv2_code"]))
    elif operation == "add_weighted":
        result = cv2.addWeighted(
            image,
            case["alpha"],
            _as_array(case["other"]),
            case["beta"],
            case["gamma"],
        )
    elif operation == "convert_scale_abs":
        result = cv2.convertScaleAbs(image, alpha=case["alpha"], beta=case["beta"])
    elif operation == "resize":
        result = cv2.resize(
            image,
            tuple(case["size"]),
            interpolation=getattr(cv2, case["interpolation"]),
        )
    elif operation == "copy_make_border":
        top, bottom, left, right = case["borders"]
        result = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=case["value"],
        )
    else:
        raise AssertionError(f"Unhandled reference operation: {operation}")

    expected = _as_array(case["expected"])
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_image_fixture_preserves_bgr_contract() -> None:
    """The deterministic PPM fixture establishes RGB-file to BGR-array order."""
    image = cv2.imread(str(IMAGE_FIXTURE_PATH), cv2.IMREAD_UNCHANGED)

    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape == (2, 2, 3)
    np.testing.assert_array_equal(
        image,
        np.asarray(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [255, 255, 255]],
            ],
            dtype=np.uint8,
        ),
    )


def test_image_format_matrix_is_explicit() -> None:
    """Required image formats and alpha behavior are explicit in the contract."""
    formats = CONTRACT["image_formats"]
    names = {item["format"] for item in formats}
    extensions = [extension for item in formats for extension in item["extensions"]]

    assert names == {"PNG", "JPEG", "BMP", "TIFF"}
    assert len(extensions) == len(set(extensions))
    assert all(item["read"] and item["write"] for item in formats)
    assert next(item for item in formats if item["format"] == "PNG")["alpha"] == (
        "preserve with unchanged flag"
    )


def test_video_matrix_guarantees_only_default_codec() -> None:
    """The video contract separates guaranteed and wheel-dependent codecs."""
    video = CONTRACT["video_formats"]
    default = video["default_codec"]

    assert default == {
        "fourcc": "mp4v",
        "encoder": "mpeg4",
        "availability": "guaranteed",
    }
    assert all(
        codec["availability"] == "capability-dependent"
        for codec in video["optional_codecs"]
    )
    assert video["audio"]["method"] == "stream remux"


def test_hershey_contract_covers_base_and_italic_faces() -> None:
    """All eight base faces and the shared italic modifier are enumerated."""
    fonts = CONTRACT["hershey_fonts"]

    assert len(fonts["base_values"]) == 8
    assert fonts["accepted_values"] == fonts["base_values"] + [
        value + fonts["italic_modifier"] for value in fonts["base_values"]
    ]


def test_spike_evidence_preserves_open_cross_platform_gates() -> None:
    """Spike evidence records its scope without overstating completion."""
    evidence = CONTRACT["spike_evidence"]

    assert evidence["pyav"]["status"] == "single-platform-evidence"
    assert evidence["pyav"]["open_gates"]
    assert evidence["hershey"]["status"] == "candidate-source-evaluated"
    assert evidence["hershey"]["open_gates"]
    assert evidence["image_arithmetic"]["open_gates"]
